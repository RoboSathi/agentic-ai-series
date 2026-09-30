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

from pydantic import BaseModel, Field

from agentic_commons.durability import (
    SUMMARY,
    DurableGraphRunner,
    ResumableRunError,
    RunOutcome,
    RunRegistry,
)
from agentic_commons.memory import MemoryFacade
from research_agent.archive.store import ResearchArchive
from research_agent.config.settings import ResearchAgentSettings
from research_agent.domain.report import Report, SessionPhase
from research_agent.domain.rubric import Contract
from research_agent.graph.run_nodes import RunNodes
from research_agent.graph.state import ScopingState
from research_agent.research.resources import RunResources
from research_agent.research.telemetry import RunTelemetry

ResourcesFactory = Callable[[Contract], RunResources]
_PHASE_KEY = "phase"
_TERMINAL: frozenset[SessionPhase] = frozenset({"DELIVERED", "CANCELLED", "FAILED"})


class TurnResult(BaseModel):
    """What one turn returned to the shell (the TUI reads `response`)."""

    response: str
    phase: SessionPhase
    run_id: str = ""
    report: Report | None = None


class SessionRecord(BaseModel):
    """Where one session stands."""

    phase: SessionPhase = "DELIVERED"
    scopes: int = 0
    scope_thread: str = ""
    run_thread: str = ""
    scope: ScopingState | None = None
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
    ) -> None:
        self._scoping = scoping
        self._runs = runs
        self._nodes = run_nodes
        self._resources = resources
        self._archive = archive
        self._telemetry = telemetry
        self._settings = settings
        self._memory = memory
        self._sessions: dict[str, SessionRecord] = {}
        self._active: RunResources | None = None

    # -- the shell's contract -------------------------------------------------------------

    @property
    def runs(self) -> RunRegistry:
        return self._runs.registry

    def session(self, session_id: str) -> SessionRecord:
        return self._sessions.setdefault(session_id, SessionRecord())

    async def handle(self, query: str, session_id: str) -> TurnResult:
        record = self.session(session_id)
        verb = _verb(query)
        names = self._settings.command_names
        if verb in (names.cancel, names.history, names.show):
            return self._control(verb, query, record)
        if record.phase == "RUNNING":
            return TurnResult(response=self._settings.messages.run_busy, phase="RUNNING")
        if record.phase in _TERMINAL:
            return await self._new_scope(query, session_id, record)
        return await self._continue_scope(query, session_id, record)

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

    async def _new_scope(self, query: str, session_id: str, record: SessionRecord) -> TurnResult:
        record.scopes += 1
        record.scope_thread = f"{session_id}-scope-{record.scopes}-{uuid.uuid4().hex[:6]}"
        record.phase = "SCOPING"
        payload = {"session_id": session_id, "first_message": query}
        outcome = await self._scoping.start(
            record.scope_thread,
            payload,
            session_id=session_id,
            subject_id=session_id,
            preview=query,
        )
        return await self._after_scope(outcome, session_id, record)

    async def _continue_scope(
        self, query: str, session_id: str, record: SessionRecord
    ) -> TurnResult:
        outcome = await self._scoping.resume(record.scope_thread, query)
        return await self._after_scope(outcome, session_id, record)

    async def _after_scope(
        self, outcome: RunOutcome, session_id: str, record: SessionRecord
    ) -> TurnResult:
        scope = ScopingState.model_validate(outcome.values)
        record.scope = scope
        self._note(session_id, "scope_phase", scope.phase)
        if outcome.status == "interrupted":
            record.phase = "RUBRIC_REVIEW" if scope.phase == "RUBRIC_REVIEW" else "SCOPING"
            reply = str((outcome.interrupt or {}).get(SUMMARY, scope.reply))
            return TurnResult(response=reply, phase=record.phase)
        if scope.phase == "APPROVED" and scope.contract is not None:
            return await self._run(scope.contract, scope.reply, session_id, record)
        record.phase = "DELIVERED"
        return TurnResult(response=scope.reply, phase=record.phase)

    # -- the autonomous run ---------------------------------------------------------------

    async def _run(
        self, contract: Contract, approved: str, session_id: str, record: SessionRecord
    ) -> TurnResult:
        resources = self._resources(contract)
        self._active = resources
        self._nodes.bind(resources)
        self._telemetry.retrace(contract.contract_id)
        record.phase, record.run_thread = "RUNNING", f"{contract.contract_id}"
        self._note(session_id, "run", contract.contract_id)
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
        return self._delivered(outcome.values, approved, session_id, record)

    def _delivered(
        self, values: Mapping[str, Any], approved: str, session_id: str, record: SessionRecord
    ) -> TurnResult:
        status = str(values.get("status") or "FAILED")
        report = values.get("report")
        record.last_report = report if isinstance(report, Report) else None
        record.run_values = dict(values)
        record.phase = "DELIVERED" if status in ("COMPLETE", "PARTIAL") else status  # type: ignore[assignment]
        self._note(session_id, "run_status", status)
        summary = (
            record.last_report.text()
            if record.last_report is not None
            else str(values.get("stop_reason", ""))
        )
        head = self._settings.messages.delivered.format(
            run_id=record.run_thread, status=status, summary=""
        )
        return TurnResult(
            response=f"{approved}\n\n{head}\n\n{summary}".strip(),
            phase=record.phase,
            run_id=record.run_thread,
            report=record.last_report,
        )

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
