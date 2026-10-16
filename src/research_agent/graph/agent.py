"""The research agent: `handle(query, session_id)` dispatches on the session's phase.

    SCOPING / RUBRIC_REVIEW   resume the scope thread with this message
    RUNNING                   the user may only watch or stop it
    STOPPED                   paused at its last checkpoint: ↻ Resume or ✕ Abort (or start anew)
    DELIVERED / CANCELLED / FAILED   a new message starts a new scope

Both graphs run durably through the platform's `DurableGraphRunner`: the
scope thread pauses on the user every turn, and `/approve` ends it with a
frozen contract, at which point the run thread starts in the same call —
so the caller (the console, or an eval's ScriptedUser) sees one turn that
began with approval and ended with a delivered report.
"""

import uuid
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

import yaml
from pydantic import BaseModel, Field

from agent_platform.config import get_settings
from agent_platform.durability import (
    SUMMARY,
    CheckpointInfo,
    DurableGraphRunner,
    ResumableRunError,
    ResumeResult,
    RunOutcome,
    RunRegistry,
)
from agent_platform.durability.stop_report import stop_report
from agent_platform.logging import get_logger
from agent_platform.memory import MemoryFacade, TurnRecorder
from agent_platform.observability.langsmith import (
    NullTraceExporter,
    TraceExporter,
    turn_trace,
)
from agent_platform.observability.types import StatusKind
from agent_platform.observability.units import (
    close_task,
    current_task_kind,
    mark_cancelled,
    open_task,
)
from agent_platform.security.pii.active import admit_if_on
from agent_platform.security.pii.source import Admitted
from research_agent.config.settings import ResearchAgentSettings
from research_agent.domain.report import COVERAGE_AND_LIMITS, Report, SessionPhase
from research_agent.domain.rubric import Contract
from research_agent.domain.scope import Choice
from research_agent.graph.answerer import DirectAnswerer
from research_agent.graph.run_nodes import RunNodes
from research_agent.graph.state import ScopingState
from research_agent.guardrails.input_screen import InputScreen
from research_agent.guardrails.refusal import RefusalRule
from research_agent.observability.telemetry import RunTelemetry
from research_agent.research.resources import RunResources
from research_agent.scoping import checklist
from research_agent.synthesis.reader_view import render_reader_view

ResourcesFactory = Callable[[Contract], RunResources]
_PHASE_KEY = "phase"
# A stopped run waits on Resume or Abort, but a new message may still start something new.
_TERMINAL: frozenset[SessionPhase] = frozenset({"DELIVERED", "CANCELLED", "FAILED", "STOPPED"})


class TurnResult(BaseModel):
    """What one turn returned to the shell (the TUI reads `response`)."""

    response: str
    phase: SessionPhase
    run_id: str = ""
    report: Report | None = None
    # Answers offered as a menu beside free text, when the question has a closed set.
    choices: list[Choice] = Field(default_factory=list)


class SessionRecord(BaseModel):
    """Where one session stands."""

    phase: SessionPhase = "DELIVERED"
    scopes: int = 0
    scope_thread: str = ""
    run_thread: str = ""
    scope: ScopingState | None = None
    # The last question asked, so a bare /research can go deeper on it.
    last_question: str = ""
    last_report: Report | None = None
    run_values: dict[str, Any] = Field(default_factory=dict)


class ResearchAgent:
    """Scoping and research, behind one `handle()`."""

    def __init__(
        self,
        scoping: DurableGraphRunner,
        runs: DurableGraphRunner,
        run_nodes: RunNodes,
        resources: ResourcesFactory,
        telemetry: RunTelemetry,
        settings: ResearchAgentSettings,
        memory: MemoryFacade | None = None,
        answerer: DirectAnswerer | None = None,
        screen: InputScreen | None = None,
        refusal: RefusalRule | None = None,
        trace_exporter: TraceExporter | None = None,
        dream_recorder: TurnRecorder | None = None,
    ) -> None:
        self._scoping = scoping
        # The person's own message, screened before any layer reads it (None: no screen declared).
        self._screen = screen
        self._runs = runs
        self._nodes = run_nodes
        self._resources = resources
        self._telemetry = telemetry
        self._logger = get_logger("research_agent")
        self._settings = settings
        self._memory = memory
        # A plain question is answered directly; without an answerer (the `direct_answer`
        # feature withheld) every question starts a research scope instead.
        self._answerer = answerer
        # Requests this agent will not help with, refused before any model reads them.
        self._refusal = refusal
        # Where each turn's trace is exported (LangSmith), if anywhere.
        self._trace_exporter = trace_exporter or NullTraceExporter()
        # Each turn's dream event, for `make dream ra` (None: no memory declared).
        self._dream_recorder = dream_recorder
        self._sessions: dict[str, SessionRecord] = {}
        self._active: RunResources | None = None
        # The run thread in flight: what Stop pauses through the durable runner.
        self._active_thread: str | None = None

    # -- the shell's contract -------------------------------------------------------------

    @property
    def runs(self) -> RunRegistry:
        return self._runs.registry

    async def checkpoint_history(self, thread_id: str) -> list[CheckpointInfo]:
        """One run thread's checkpoints, newest first (the `DurableAgent` contract)."""
        return await self._runs.history(thread_id, get_settings().commands.default_limit)

    def session(self, session_id: str) -> SessionRecord:
        return self._sessions.setdefault(session_id, SessionRecord())

    async def handle(self, query: str, session_id: str) -> TurnResult:
        """One turn, under a trace of its own that every log line it causes carries.

        The message becomes tokens before anything holds it (`security.pii.source`).
        """
        admitted = admit_if_on(query)
        query = admitted.text
        with turn_trace(
            self._trace_exporter,
            agent=get_settings().agent.id,
            session=session_id,
            inputs={"query": query},
        ) as trace:
            result = await self._summarised(admitted, session_id)
            self._record_turn(query, session_id, result)
            trace.annotate({"phase": result.phase, "run_id": result.run_id})
            trace.finish({"response": result.response, "phase": result.phase})
            return result

    async def _summarised(self, admitted: Admitted, session_id: str) -> TurnResult:
        """The turn, ended by the same TURN SUMMARY card csa's turns end with."""
        reporter = self._telemetry.reporter
        if reporter is None:
            return await self._handle(admitted.text, session_id, admitted)
        words = self._settings.turn_summary
        reporter.begin_turn(session=session_id)
        try:
            result = await self._handle(admitted.text, session_id, admitted)
        except Exception:
            reporter.end_turn("failed", words.crashed)
            raise
        status: StatusKind = (
            "waiting"
            if result.phase in words.waiting
            else "failed"
            if result.phase in words.failed
            else "completed"
        )
        reporter.end_turn(status, words.outcomes.get(result.phase, result.phase))
        return result

    async def _handle(
        self, query: str, session_id: str, admitted: Admitted | None = None
    ) -> TurnResult:
        record = self.session(session_id)
        verb = _verb(query)
        names = self._settings.command_names
        if verb == names.cancel:
            return self._control(verb, query, record)
        if record.phase == "RUNNING":
            return TurnResult(response=self._settings.messages.run_busy, phase="RUNNING")
        research = verb == names.research
        if self._screen is not None and (verb is None or research):
            screened = await self._screen.screen(query, session_id, admitted)
            if screened.blocked:
                return TurnResult(response=screened.refusal, phase=record.phase)
            query = screened.text
        if research:
            return await self._research(query, session_id, record)
        if record.phase in _TERMINAL:
            return await self._open_turn(query, session_id, record)
        return await self._continue_scope(query, session_id, record)

    async def _open_turn(self, query: str, session_id: str, record: SessionRecord) -> TurnResult:
        """A plain question: answered directly (with tools on demand), unless it is refused.

        `/research` is the only way into a research run; nothing guesses a user into one.
        """
        if self._refusal is not None and self._refusal.refused(query):
            self._telemetry.audit("answer.refused", "deny", "matches a refusal pattern")
            close_task(self._settings.tasks.refused)
            return TurnResult(response=self._settings.messages.refuse, phase=record.phase)
        if self._answerer is None:
            return await self._new_scope(query, session_id, record)
        return await self._answer(query, session_id, record)

    async def _answer(self, query: str, session_id: str, record: SessionRecord) -> TurnResult:
        """A direct answer: the ReAct loop over the read-only tools, with no scope."""
        assert self._answerer is not None
        record.last_question = query
        self._telemetry.retrace(f"answer-{uuid.uuid4().hex[:8]}")
        tasks = self._settings.tasks
        open_task(tasks.direct_answer)
        answer = await self._answerer.respond(query, session_id)
        close_task(tasks.answer_failed if answer.failed else tasks.answered)
        return TurnResult(response=answer.text, phase=record.phase)

    async def _research(self, query: str, session_id: str, record: SessionRecord) -> TurnResult:
        """`/research [question]` starts a research scope.

        A bare `/research` goes deeper on the last question asked; an
        interview in progress is abandoned for the new one.
        """
        parts = query.strip().split(maxsplit=1)
        question = parts[1].strip() if len(parts) > 1 else record.last_question
        if not question:
            return TurnResult(response=self._settings.messages.new_scope, phase=record.phase)
        return await self._new_scope(question, session_id, record)

    @property
    def running(self) -> bool:
        """Whether a research run is in flight (what the Stop button stops)."""
        return self._active_thread is not None

    def cancel(self) -> bool:
        """Stop the run in progress at once; it pauses at its last checkpoint, resumable.

        Paused through the durable runner, as every agent's run is. The run's own flag is set
        too, so a stop that lands while the runner cannot cancel still ends it at the next
        safe point instead of being lost.
        """
        if self._active is None or self._active.cancelled:
            return False
        self._active.cancel.set()
        paused = bool(self._active_thread and self._runs.cancel(self._active_thread))
        self._telemetry.audit("run.cancel_requested", "allow", "operator stopped the run")
        return paused or self._active.cancelled

    def phase(self, session_id: str) -> str:
        """Where the session stands (SCOPING, RUBRIC_REVIEW, RUNNING …): it picks the buttons."""
        return self.session(session_id).phase

    # -- scoping --------------------------------------------------------------------------

    async def _new_scope(self, query: str, session_id: str, record: SessionRecord) -> TurnResult:
        record.scopes += 1
        record.scope_thread = f"{session_id}-scope-{record.scopes}-{uuid.uuid4().hex[:6]}"
        record.phase = "SCOPING"
        record.last_question = query
        # The research run is one task from the first scoping question to its report.
        open_task(self._settings.tasks.research_run)
        payload = {"session_id": session_id, "first_message": query}
        try:
            outcome = await self._scoping.start(
                record.scope_thread,
                payload,
                session_id=session_id,
                subject_id=session_id,
                preview=query,
            )
        except ResumableRunError as exc:
            return self._scope_failed(exc, record)
        return await self._after_scope(outcome, session_id, record, query)

    async def _continue_scope(
        self, query: str, session_id: str, record: SessionRecord
    ) -> TurnResult:
        try:
            outcome = await self._scoping.resume(record.scope_thread, query)
        except ResumableRunError as exc:
            return self._scope_failed(exc, record)
        return await self._after_scope(outcome, session_id, record, query)

    def _scope_failed(self, exc: ResumableRunError, record: SessionRecord) -> TurnResult:
        """A scoping step failed (a model that stalled, a provider error): say so, and start clean.

        The session ends the turn in a terminal phase, so the next message begins a new scope
        instead of resuming a thread whose step failed.
        """
        record.phase = "FAILED"
        self._close_run(self._settings.tasks.run_status.get("FAILED", ""))
        self._telemetry.audit("scope.failed", "deny", str(exc), node=exc.node, error=exc.error)
        self._logger.error("Scoping failed", node=exc.node, error=exc.error)
        return TurnResult(
            response=self._settings.messages.scope_failed.format(error=exc.error), phase="FAILED"
        )

    async def _after_scope(
        self, outcome: RunOutcome, session_id: str, record: SessionRecord, query: str
    ) -> TurnResult:
        scope = ScopingState.model_validate(outcome.values)
        record.scope = scope
        self._note(session_id, "scope_phase", scope.phase)
        if outcome.status == "interrupted":
            record.phase = "RUBRIC_REVIEW" if scope.phase == "RUBRIC_REVIEW" else "SCOPING"
            reply = str((outcome.interrupt or {}).get(SUMMARY, scope.reply))
            # The interview is part of the conversation: a direct answer after it can refer
            # back to what was asked and said.
            self._remember_turn(session_id, query, reply)
            return TurnResult(response=reply, phase=record.phase, choices=self._choices(scope))
        if scope.phase == "APPROVED" and scope.contract is not None:
            return await self._run(scope.contract, session_id, record)
        record.phase = "DELIVERED"
        # Scoping ended without a contract to run: the request was not taken on.
        self._close_run(self._settings.tasks.scope_refused)
        return TurnResult(response=scope.reply, phase=record.phase)

    def _choices(self, scope: ScopingState) -> list[Choice]:
        """The menu beside the reply: the asked item's own choices."""
        asked = checklist.item(scope.items, scope.asked_item_id)
        return list(asked.choices) if scope.phase == "INTERVIEW" and asked is not None else []

    # -- the autonomous run ---------------------------------------------------------------

    async def _run(self, contract: Contract, session_id: str, record: SessionRecord) -> TurnResult:
        record.phase, record.run_thread = "RUNNING", f"{contract.contract_id}"
        self._note(session_id, "run", contract.contract_id)
        # The ids and hash are for the log and the audit trail; the screen says what to do.
        self._logger.info(
            "Contract frozen; research starting",
            contract_id=contract.contract_id,
            contract_hash=contract.contract_hash,
            topic=contract.brief.topic,
            depth=contract.brief.depth,
        )
        return await self._drive(
            self._resources(contract),
            session_id,
            record,
            lambda: self._runs.start(
                record.run_thread,
                {"contract": contract},
                session_id=session_id,
                subject_id=contract.contract_id,
                preview=contract.brief.topic,
            ),
        )

    async def _drive(
        self,
        resources: RunResources,
        session_id: str,
        record: SessionRecord,
        run: Callable[[], Awaitable[RunOutcome]],
    ) -> TurnResult:
        """Run (or resume) the research thread with its resources bound; say how it ended."""
        self._active, self._active_thread = resources, record.run_thread
        self._nodes.bind(resources)
        self._telemetry.retrace(resources.contract.contract_id)
        record.phase = "RUNNING"
        try:
            outcome = await run()
        except ResumableRunError as exc:
            # The checkpoint holds every finished node; the operator can resume it.
            record.phase = "FAILED"
            return TurnResult(response=str(exc), phase="FAILED", run_id=record.run_thread)
        finally:
            self._active, self._active_thread = None, None
        if outcome.status == "cancelled":
            return self._paused(outcome, record)
        return self._delivered(outcome.values, session_id, record)

    def _paused(self, outcome: RunOutcome, record: SessionRecord) -> TurnResult:
        """Stopped at its last checkpoint: where, how far it got, and the two ways on."""
        record.phase = "STOPPED"
        # The turn was stopped by the operator; the run's task stays open, resumable.
        mark_cancelled()
        step = outcome.next_nodes[0] if outcome.next_nodes else ""
        self._telemetry.audit("run.stopped", "allow", "operator stopped the run", node=step)
        return TurnResult(
            response=stop_report(step, self.describe_progress(outcome.values)),
            phase="STOPPED",
            run_id=record.run_thread,
        )

    async def resume(self, thread_id: str, decision: Any | None = None) -> ResumeResult:
        """Continue a stopped run from its checkpoint, its ledger, budget and trace restored."""
        session_id, record = self._owner(thread_id)
        with turn_trace(
            self._trace_exporter,
            agent=get_settings().agent.id,
            session=session_id,
            inputs={"resume": thread_id},
            message=False,
        ) as trace:
            result = await self._resumed(thread_id, decision, session_id, record)
            trace.finish({"response": result.response, "status": result.status})
            return result

    async def _resumed(
        self, thread_id: str, decision: Any | None, session_id: str, record: SessionRecord
    ) -> ResumeResult:
        values = await self._runs.values(thread_id)
        resources = self._resources(values["contract"])
        if values.get("ledger") is not None:
            resources.ledger.restore(values["ledger"])
        resources.budget.restore(values.get("spent") or {})
        if values.get("loop_state") is not None:
            resources.governor.restore(values["loop_state"])
        if values.get("trace") is not None:
            resources.trace = values["trace"]
            # The rescue round's grant is on top of the size's caps, which a resume rebuilds.
            for counter, amount in resources.trace.rescue_grant.items():
                resources.budget.extend(counter, amount)
        result = await self._drive(
            resources,
            session_id,
            record,
            lambda: self._runs.resume(thread_id, decision, self._telemetry.reporter),
        )
        return ResumeResult(
            thread_id=thread_id,
            session_id=session_id,
            status="cancelled" if result.phase == "STOPPED" else "completed",
            response=result.response,
        )

    async def abort(self, thread_id: str) -> ResumeResult:
        """End a stopped run for good: its record is kept; the session can start afresh."""
        session_id, record = self._owner(thread_id)
        with turn_trace(
            self._trace_exporter,
            agent=get_settings().agent.id,
            session=session_id,
            inputs={"abort": thread_id},
            message=False,
        ):
            outcome = await self._runs.abort(thread_id, self._telemetry.reporter)
            record.phase = "CANCELLED"
            self._close_run(self._settings.tasks.run_status.get("CANCELLED", ""))
            self._telemetry.audit("run.aborted", "allow", "operator aborted the stopped run")
        return ResumeResult(
            thread_id=thread_id,
            session_id=session_id,
            status=outcome.status,
            response=get_settings().durability.stop_report.aborted,
        )

    def describe_progress(self, values: dict[str, Any]) -> dict[str, Any]:
        """How far a run got: its topic, round, sub-questions, evidence and budget spent."""
        words = self._settings.tui.cards
        contract = values.get("contract")
        ledger = values.get("ledger")
        entries = list(getattr(ledger, "entries", []) or [])
        nodes = [node for plan in values.get("plans", []) for node in plan.nodes]
        answered = {entry.subject for entry in entries}
        spent = values.get("spent") or {}
        return {
            words["stop.topic"]: contract.brief.topic if isinstance(contract, Contract) else "",
            words["stop.round"]: words["stop.round_value"].format(
                round=values.get("round", 0), gap_fill=values.get("gap_fill_round", 0)
            ),
            words["stop.questions"]: words["stop.questions_value"].format(
                answered=len({n.node_id for n in nodes} & answered), asked=len(nodes)
            ),
            words["stop.evidence"]: words["stop.evidence_value"].format(
                items=len(entries), sources=len({e.source_url for e in entries})
            ),
            words["stop.spent"]: {name: f"{value:g}" for name, value in spent.items()},
        }

    def _owner(self, thread_id: str) -> tuple[str, SessionRecord]:
        """The session whose run this thread is (its record, created if the process restarted)."""
        run = self._runs.registry.get(thread_id)
        session_id = run.session_id if run is not None else ""
        record = self.session(session_id)
        record.run_thread = thread_id
        return session_id, record

    def _delivered(
        self, values: Mapping[str, Any], session_id: str, record: SessionRecord
    ) -> TurnResult:
        status = str(values.get("status") or "FAILED")
        report = values.get("report")
        record.last_report = report if isinstance(report, Report) else None
        record.run_values = dict(values)
        record.phase = "DELIVERED" if status in ("COMPLETE", "PARTIAL") else status  # type: ignore[assignment]
        self._close_run(self._settings.tasks.run_status.get(status, ""))
        self._note(session_id, "run_status", status)
        summary = (
            self._reader_text(record.last_report)
            if record.last_report is not None
            else str(values.get("stop_reason", ""))
        )
        self._logger.info("Research run finished", run_id=record.run_thread, status=status)
        # The report speaks for itself; a run with no report says how it ended.
        head = (
            ""
            if record.last_report is not None
            else self._settings.messages.delivered.format(
                run_id=record.run_thread, status=status, summary=""
            )
        )
        self._remember_run(session_id, record, status)
        return TurnResult(
            response="\n\n".join(part for part in (head, summary) if part).strip(),
            phase=record.phase,
            run_id=record.run_thread,
            report=record.last_report,
        )

    def _close_run(self, outcome: str) -> None:
        """Close the research run's task with `outcome` (a run resumed after its task was
        abandoned — the session moved on meanwhile — reopens it first, so the ending counts)."""
        if not outcome:
            return
        kind = self._settings.tasks.research_run
        if current_task_kind() != kind:
            open_task(kind)
        close_task(outcome)

    def _remember_run(self, session_id: str, record: SessionRecord, status: str) -> None:
        """Working memory keeps a pointer to the run, so a follow-up has something to refer to."""
        note = self._settings.messages.run_note.format(
            run_id=record.run_thread, status=status, topic=record.last_question
        )
        self._remember_turn(session_id, record.last_question, note)

    def _record_turn(self, query: str, session_id: str, result: TurnResult) -> None:
        """Leave the turn in the dream inbox; the agent's own work has no subject."""
        if self._dream_recorder is not None:
            self._dream_recorder.record(
                session_id=session_id, subject_id="", asked=query, answered=result.response
            )

    def _remember_turn(self, session_id: str, said: str, replied: str) -> None:
        """One exchange into the session's working memory (when the agent keeps one)."""
        if self._memory is not None and self._memory.working is not None:
            self._memory.working.get(session_id).add_history(said, replied)

    def _reader_text(self, report: Report) -> str:
        """What the person reads: numbered sources and one finding per line (or the raw text)."""
        if not self._settings.reader_view:
            return report.text()
        return render_reader_view(
            report,
            citation_pattern=self._settings.synthesis.citation_pattern,
            hidden_headings=self._hidden_headings(),
            wording=self._settings.reader_view_wording,
            source_labels=self._settings.research.source_labels,
        )

    def _hidden_headings(self) -> frozenset[str]:
        """Sections written in code, not by the model; the reader view shows them its own way."""
        templates = yaml.safe_load(self._settings.synthesis.resolved_templates_path.read_text())
        return frozenset({*templates["appended"].values(), COVERAGE_AND_LIMITS})

    # -- control commands -----------------------------------------------------------------

    def _control(self, verb: str | None, query: str, record: SessionRecord) -> TurnResult:
        messages = self._settings.messages
        reply = messages.cancel_requested if self.cancel() else messages.no_active_run
        return TurnResult(response=reply, phase=record.phase)

    def _note(self, session_id: str, key: str, value: str) -> None:
        """Working memory holds the live run state of this session."""
        if self._memory is not None and self._memory.working is not None:
            self._memory.note(key, value, session_id, source="research_agent")


def _verb(line: str) -> str | None:
    stripped = line.strip()
    return (
        stripped[1:].split(maxsplit=1)[0].casefold()
        if stripped.startswith("/") and len(stripped) > 1
        else None
    )
