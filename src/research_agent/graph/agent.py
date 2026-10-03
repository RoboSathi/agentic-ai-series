"""The research agent: `handle(query, session_id)` dispatches on the session's phase.

    SCOPING / RUBRIC_REVIEW   resume the scope thread with this message
    RUNNING                   the user may only watch or cancel
    DELIVERED / CANCELLED / FAILED   a new message starts a new scope

Both graphs run durably through the platform's `DurableGraphRunner`: the
scope thread pauses on the user every turn, and `/approve` ends it with a
frozen contract, at which point the run thread starts in the same call —
so the caller (the console, or an eval's ScriptedUser) sees one turn that
began with approval and ended with a delivered report.
"""

import uuid
from collections.abc import Callable, Mapping
from typing import Any

import yaml
from pydantic import BaseModel, Field

from agent_platform.config import get_settings
from agent_platform.durability import (
    SUMMARY,
    DurableGraphRunner,
    ResumableRunError,
    RunOutcome,
    RunRegistry,
)
from agent_platform.logging import bound_log_context, get_logger, new_trace_id
from agent_platform.memory import MemoryFacade
from research_agent.archive.store import ResearchArchive
from research_agent.config.settings import ResearchAgentSettings
from research_agent.domain.report import COVERAGE_AND_LIMITS, Report, SessionPhase
from research_agent.domain.rubric import Contract
from research_agent.domain.scope import QA, Choice, Mode
from research_agent.graph import cards
from research_agent.graph.answerer import DirectAnswerer
from research_agent.graph.run_nodes import RunNodes
from research_agent.graph.state import ScopingState
from research_agent.guardrails.input_screen import InputScreen
from research_agent.intent import ResearchIntentRouter
from research_agent.observability.telemetry import RunTelemetry
from research_agent.research.resources import RunResources
from research_agent.scoping import checklist
from research_agent.synthesis.reader_view import render_reader_view

ResourcesFactory = Callable[[Contract], RunResources]
_PHASE_KEY = "phase"
_TERMINAL: frozenset[SessionPhase] = frozenset({"DELIVERED", "CANCELLED", "FAILED"})
# The front door's card: the platform's INTENT stage, its outcome by where the message went.
INTENT_STAGE = "intent_classification"
_ROUTE_STATUS: Mapping[str, Any] = {
    "answer": "completed",
    "research": "completed",
    "refuse": "blocked",
}


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
        archive: ResearchArchive,
        telemetry: RunTelemetry,
        settings: ResearchAgentSettings,
        memory: MemoryFacade | None = None,
        router: ResearchIntentRouter | None = None,
        answerer: DirectAnswerer | None = None,
        screen: InputScreen | None = None,
    ) -> None:
        self._scoping = scoping
        # The person's own message, screened before any layer reads it (None: no screen declared).
        self._screen = screen
        self._runs = runs
        self._nodes = run_nodes
        self._resources = resources
        self._archive = archive
        self._telemetry = telemetry
        self._logger = get_logger("research_agent")
        self._settings = settings
        self._memory = memory
        # Both or neither: without them every question starts a scope (the `direct_answer`
        # feature withheld, or the offline fixture posture).
        self._router = router if answerer is not None else None
        self._answerer = answerer
        self._sessions: dict[str, SessionRecord] = {}
        self._active: RunResources | None = None

    # -- the shell's contract -------------------------------------------------------------

    @property
    def runs(self) -> RunRegistry:
        return self._runs.registry

    def session(self, session_id: str) -> SessionRecord:
        return self._sessions.setdefault(session_id, SessionRecord())

    async def handle(self, query: str, session_id: str) -> TurnResult:
        """One turn, under a trace of its own that every log line it causes carries."""
        with bound_log_context(
            agent=get_settings().agent.id, session=session_id, trace_id=new_trace_id()
        ):
            return await self._handle(query, session_id)

    async def _handle(self, query: str, session_id: str) -> TurnResult:
        record = self.session(session_id)
        verb = _verb(query)
        names = self._settings.command_names
        if verb in (names.cancel, names.history, names.show):
            return self._control(verb, query, record)
        if record.phase == "RUNNING":
            return TurnResult(response=self._settings.messages.run_busy, phase="RUNNING")
        mode = self._forced_mode(verb)
        if self._screen is not None and (verb is None or mode is not None):
            screened = await self._screen.screen(query)
            if screened.blocked:
                return TurnResult(
                    response=self._settings.messages.input_blocked, phase=record.phase
                )
            query = screened.text
        if mode is not None:
            return await self._forced_scope(mode, query, session_id, record)
        if record.phase in _TERMINAL:
            return await self._open_turn(query, session_id, record)
        return await self._continue_scope(query, session_id, record)

    async def _open_turn(self, query: str, session_id: str, record: SessionRecord) -> TurnResult:
        """A message that starts something new: answered directly, researched, or refused."""
        if self._router is None or self._answerer is None:
            return await self._new_scope(query, session_id, record)
        route = await self._router.route(query)
        self._telemetry.card(
            INTENT_STAGE, *cards.intent_card(route), status=_ROUTE_STATUS[route.label]
        )
        if route.label == "research":
            return await self._new_scope(query, session_id, record, forced="research")
        if route.label == "refuse":
            return TurnResult(response=self._settings.messages.refuse, phase=record.phase)
        return await self._answer(query, session_id, record)

    async def _answer(self, query: str, session_id: str, record: SessionRecord) -> TurnResult:
        """A direct answer: the ReAct loop over the read-only tools, with no scope."""
        assert self._answerer is not None
        record.last_question = query
        self._telemetry.retrace(f"answer-{uuid.uuid4().hex[:8]}")
        return TurnResult(
            response=await self._answerer.answer(query, session_id), phase=record.phase
        )

    def _forced_mode(self, verb: str | None) -> Mode | None:
        names = self._settings.command_names
        return "qa" if verb == names.qa else "research" if verb == names.research else None

    async def _forced_scope(
        self, mode: Mode, query: str, session_id: str, record: SessionRecord
    ) -> TurnResult:
        """`/qa <question>` answers directly; `/research [question]` starts a research scope.

        A bare `/research` goes deeper on the last question asked; an
        interview in progress is abandoned for the new one.
        """
        parts = query.strip().split(maxsplit=1)
        question = parts[1].strip() if len(parts) > 1 else record.last_question
        if not question:
            return TurnResult(response=self._settings.messages.new_scope, phase=record.phase)
        if mode == "qa" and self._answerer is not None:
            # A quick cited answer, no interview: the direct answerer, not a scoped run.
            return await self._answer(question, session_id, record)
        return await self._new_scope(question, session_id, record, forced=mode)

    def cancel(self) -> bool:
        """Cancel the run in progress, if any (it stops at the next safe point)."""
        if self._active is None or self._active.cancelled:
            return False
        self._active.cancel.set()
        self._telemetry.audit("run.cancel_requested", "allow", "operator cancelled the run")
        return True

    def panel_state(self, session_id: str) -> dict[str, Any]:
        """What the side panel shows: the checklist, then the run's meters."""
        record = self.session(session_id)
        state: dict[str, Any] = {"phase": record.phase}
        if record.scope is not None:
            state["items"] = [(i.item_id, i.status) for i in record.scope.items]
        if self._active is not None:
            state["budget"] = {
                k: (c.spent, c.cap) for k, c in self._active.budget.snapshot().items()
            }
            state["ledger"] = len(self._active.ledger.entries)
            state["rejected"] = len(self._active.trace.rejected_sources)
        return state

    def history(self, limit: int) -> list[dict[str, Any]]:
        return [row.model_dump() for row in self._archive.history(limit)]

    def show(self, run_id: str) -> str | None:
        return self._archive.report(run_id)

    # -- scoping --------------------------------------------------------------------------

    async def _new_scope(
        self, query: str, session_id: str, record: SessionRecord, forced: Mode | None = None
    ) -> TurnResult:
        record.scopes += 1
        record.scope_thread = f"{session_id}-scope-{record.scopes}-{uuid.uuid4().hex[:6]}"
        record.phase = "SCOPING"
        record.last_question = query
        payload = {"session_id": session_id, "first_message": query, "forced_mode": forced}
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
        return TurnResult(response=scope.reply, phase=record.phase)

    def _choices(self, scope: ScopingState) -> list[Choice]:
        """The menu beside the reply: the reframe options, Approve or Done when either is the
        next step, or the asked item's own choices."""
        if scope.phase == "REFRAME":
            options = self._settings.scoping.reframe_options
            return [Choice(label=option.label, value=option.mode) for option in options]
        notes, names = self._settings.messages, self._settings.command_names
        if scope.phase == "RUBRIC_REVIEW":
            return [Choice(label=notes.approve_choice, value=f"/{names.approve}")]
        if scope.phase == "INTERVIEW" and scope.proposed:
            return [Choice(label=notes.done_choice, value=f"/{names.done}")]
        asked = checklist.item(scope.items, scope.asked_item_id)
        return list(asked.choices) if scope.phase == "INTERVIEW" and asked is not None else []

    # -- the autonomous run ---------------------------------------------------------------

    async def _run(self, contract: Contract, session_id: str, record: SessionRecord) -> TurnResult:
        resources = self._resources(contract)
        self._active = resources
        self._nodes.bind(resources)
        self._telemetry.retrace(contract.contract_id)
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
        try:
            outcome = await self._runs.start(
                record.run_thread,
                {"contract": contract},
                session_id=session_id,
                subject_id=contract.contract_id,
                preview=contract.brief.topic,
            )
        except ResumableRunError as exc:
            # The checkpoint holds every finished node; the operator can resume it.
            record.phase = "FAILED"
            return TurnResult(response=str(exc), phase="FAILED", run_id=record.run_thread)
        finally:
            self._active = None
        return self._delivered(outcome.values, session_id, record)

    def _delivered(
        self, values: Mapping[str, Any], session_id: str, record: SessionRecord
    ) -> TurnResult:
        status = str(values.get("status") or "FAILED")
        report = values.get("report")
        record.last_report = report if isinstance(report, Report) else None
        record.run_values = dict(values)
        record.phase = "DELIVERED" if status in ("COMPLETE", "PARTIAL") else status  # type: ignore[assignment]
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
        contract = values.get("contract")
        quick = isinstance(contract, Contract) and contract.brief.archetype == QA
        tail = self._settings.messages.qa_followup if quick else ""
        self._remember_run(session_id, record, status)
        return TurnResult(
            response="\n\n".join(part for part in (head, summary, tail) if part).strip(),
            phase=record.phase,
            run_id=record.run_thread,
            report=record.last_report,
        )

    def _remember_run(self, session_id: str, record: SessionRecord, status: str) -> None:
        """Working memory keeps a pointer to the run, so a follow-up has something to refer to."""
        note = self._settings.messages.run_note.format(
            run_id=record.run_thread, status=status, topic=record.last_question
        )
        self._remember_turn(session_id, record.last_question, note)

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
        """Sections written in code for the archive; the reader view shows them its own way."""
        templates = yaml.safe_load(self._settings.synthesis.resolved_templates_path.read_text())
        return frozenset({*templates["appended"].values(), COVERAGE_AND_LIMITS})

    # -- control commands -----------------------------------------------------------------

    def _control(self, verb: str | None, query: str, record: SessionRecord) -> TurnResult:
        names, messages = self._settings.command_names, self._settings.messages
        if verb == names.cancel:
            reply = messages.cancel_requested if self.cancel() else messages.no_active_run
            return TurnResult(response=reply, phase=record.phase)
        if verb == names.history:
            rows = self.history(self._settings.archive.history_limit)
            lines = [
                f"{r['run_id']}  {r['status']:<8} {r['archetype']:<18} {r['topic']}" for r in rows
            ]
            return TurnResult(
                response="\n".join(lines) or messages.history_empty, phase=record.phase
            )
        run_id = query.split(maxsplit=1)[1].strip() if len(query.split()) > 1 else ""
        shown = self.show(run_id)
        return TurnResult(
            response=shown or messages.unknown_run.format(run_id=run_id), phase=record.phase
        )

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
