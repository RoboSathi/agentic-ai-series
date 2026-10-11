"""The single tool path: every tool call a model proposes, governed and recorded once.

ADR 0009 D9; schema §6.2. A proposal goes, in order, through:

1. the registry — an unknown tool, or arguments that fail its schema, stop here;
2. the turn's loop control — an identical call already made is suppressed;
3. the Gatekeeper, when the agent has one — it rules on authority (and on
   whether the tool was offered), and writes its own audit entry; when it is
   withheld the call proceeds UNGOVERNED, and that is audited here;
4. the Executor — deadline and idempotency, always.

Wherever the proposal stops, one `fact_tool_call` row says which stage it
reached (UL "Tool proposal"), so tool-call validity, duplicate rate and tool
error rate are measured the same way for every agent. The caller keeps what is
its own: the cards it draws, the authority context it builds, what it tells
the model. It reads all of that off the returned `ToolOutcome`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final, Literal, Protocol, cast, runtime_checkable

from pydantic import BaseModel, ConfigDict

from agent_platform.audit import STANDING_FIELD, AuditEntry, AuditLedger
from agent_platform.audit.types import Decision
from agent_platform.executor import ApprovedToolCall, ExecutionOutcome, ToolExecutor
from agent_platform.gatekeeper import Gatekeeper
from agent_platform.gatekeeper.types import AuthorityContext, AuthorityDecision
from agent_platform.llm.types import ModelToolCall
from agent_platform.logging import get_current_trace_id, get_logger
from agent_platform.loop_control.types import TOOL_STAGE, LoopVerdict
from agent_platform.observability.facts import ToolCallFact, fact_store
from agent_platform.tools.registry import NamespacedToolRegistry
from agent_platform.tools.types import ToolCall

ToolStage = Literal["invalid", "unknown", "not_offered", "refused", "duplicate", "executed"]
INVALID: Final[ToolStage] = "invalid"
UNKNOWN: Final[ToolStage] = "unknown"
NOT_OFFERED: Final[ToolStage] = "not_offered"
REFUSED: Final[ToolStage] = "refused"
DUPLICATE: Final[ToolStage] = "duplicate"
EXECUTED: Final[ToolStage] = "executed"

# The Gatekeeper question that rules on whether a tool was offered this turn.
_OFFERED_QUESTION = "offered"
_CONTINUE = "continue"
_ALLOW: Final[Decision] = "allow"


@runtime_checkable
class ActionGate(Protocol):
    """The loop's say over each action (`LoopGovernor`): run it or not, and what came of it."""

    async def before_action(self, call: ToolCall) -> LoopVerdict: ...

    async def after_action(self, call: ToolCall, outcome: str, succeeded: bool) -> None: ...

    def record_refusal(self, call: ToolCall, reason: str) -> None: ...

    def forget(self, call: ToolCall) -> None: ...


class ToolTurn(BaseModel):
    """What the caller owns about the turn a proposal is made in."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    loop: ActionGate
    # The authority context for this call (who asks, what was offered, mandates).
    authority: Callable[[ToolCall], AuthorityContext]
    # A last transformation of the validated call (e.g. PII minimisation of its params).
    prepare: Callable[[ToolCall], ToolCall] | None = None
    # Told just before the Gatekeeper rules, and with its ruling, so the caller can
    # draw its cards in the order things happen (the Executor draws its own).
    on_authorizing: Callable[[ToolCall], None] | None = None
    on_decision: Callable[[ToolCall, AuthorityDecision], None] | None = None
    # Whether an identical retry of a call that failed is suppressed too (a read whose
    # failure will not change), or allowed (a write worth one more try).
    remember_failures: bool = False


class ToolOutcome(BaseModel):
    """How far one proposal got, and everything the caller needs to say so."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    stage: ToolStage
    tool: str
    reason: str = ""
    call: ToolCall | None = None
    control: LoopVerdict | None = None
    authority: AuthorityContext | None = None
    decision: AuthorityDecision | None = None
    execution: ExecutionOutcome | None = None

    @property
    def succeeded(self) -> bool:
        """Executed, and the tool reported success."""
        return self.execution is not None and self.execution.result.success


class ToolPath:
    """Registry → loop control → Gatekeeper (optional) → Executor, recorded at every stop."""

    def __init__(
        self,
        registry: NamespacedToolRegistry,
        executor: ToolExecutor,
        audit: AuditLedger,
        gatekeeper: Gatekeeper | None = None,
    ) -> None:
        self._registry = registry
        self._executor = executor
        self._audit = audit
        self._gatekeeper = gatekeeper
        self._logger = get_logger("tools")

    @property
    def governed(self) -> bool:
        """Whether a Gatekeeper rules on calls (False: they run UNGOVERNED)."""
        return self._gatekeeper is not None

    async def run(self, proposal: ModelToolCall, turn: ToolTurn) -> ToolOutcome:
        """Take one proposal as far as it may go; record and return where it stopped."""
        outcome = await self._run(proposal, turn)
        self._record(outcome)
        return outcome

    async def _run(self, proposal: ModelToolCall, turn: ToolTurn) -> ToolOutcome:
        resolution = self._registry.resolve(proposal.name, proposal.arguments)
        if resolution.call is None:
            stage = UNKNOWN if resolution.unknown else INVALID
            return ToolOutcome(stage=stage, tool=proposal.name, reason="; ".join(resolution.errors))
        call = turn.prepare(resolution.call) if turn.prepare else resolution.call
        control = await turn.loop.before_action(call)
        if control.decision != _CONTINUE:
            stage = cast(ToolStage, TOOL_STAGE[control.decision])
            return ToolOutcome(
                stage=stage, tool=call.action, reason=control.reason, call=call, control=control
            )
        authority = turn.authority(call)
        decided = await self._authorize(call, authority, turn)
        if isinstance(decided, ToolOutcome):
            return decided.model_copy(update={"control": control})
        execution = await self._executor.execute(decided)
        await self._settle(call, authority, execution, turn)
        return ToolOutcome(
            stage=EXECUTED,
            tool=call.action,
            reason="" if execution.result.success else execution.result.error or "",
            call=call,
            control=control,
            authority=authority,
            decision=decided.decision,
            execution=execution,
        )

    async def _authorize(
        self, call: ToolCall, authority: AuthorityContext, turn: ToolTurn
    ) -> ApprovedToolCall | ToolOutcome:
        """The Gatekeeper's ruling, or the audited UNGOVERNED approval when it is withheld."""
        if self._gatekeeper is None:
            return self._ungoverned(call, authority)
        if turn.on_authorizing is not None:
            turn.on_authorizing(call)
        decision = await self._gatekeeper.evaluate(call, authority)
        if turn.on_decision is not None:
            turn.on_decision(call, decision)
        if decision.allowed:
            return ApprovedToolCall.from_decision(call, decision, authority)
        turn.loop.record_refusal(call, decision.reason)
        stage = NOT_OFFERED if decision.question == _OFFERED_QUESTION else REFUSED
        return ToolOutcome(
            stage=stage,
            tool=call.action,
            reason=decision.reason,
            call=call,
            authority=authority,
            decision=decision,
        )

    def _ungoverned(self, call: ToolCall, authority: AuthorityContext) -> ApprovedToolCall:
        """No decision stands behind this call, and the audit trail says so."""
        approved = ApprovedToolCall.ungoverned(call, authority)
        self._audit.record(
            AuditEntry(
                trace_id=get_current_trace_id() or "",
                actor=authority.actor,
                action=call.action,
                authority_context={
                    **authority.model_dump(mode="json"),
                    STANDING_FIELD: approved.decision.standing,
                },
                decision=_ALLOW,
                rationale=approved.decision.reason,
            )
        )
        return approved

    async def _settle(
        self,
        call: ToolCall,
        authority: AuthorityContext,
        execution: ExecutionOutcome,
        turn: ToolTurn,
    ) -> None:
        """A real success is remembered (and charged to the budget); a replay is not charged."""
        if not execution.result.success:
            if turn.remember_failures:
                await turn.loop.after_action(call, execution.result.error or "", False)
            else:
                turn.loop.forget(call)
            return
        await turn.loop.after_action(call, execution.result.output, True)
        if self._gatekeeper is not None and not execution.replayed:
            self._gatekeeper.commit(call, authority)

    @staticmethod
    def _record(outcome: ToolOutcome) -> None:
        execution = outcome.execution
        fact_store().append(
            ToolCallFact(
                tool=outcome.tool,
                stage=outcome.stage,
                succeeded=execution.result.success if execution else None,
                timed_out=execution.timed_out if execution else None,
                duration_ms=execution.duration_ms if execution else None,
                side_effecting=outcome.call.side_effecting if outcome.call else None,
                reason=outcome.reason,
            )
        )


__all__ = [
    "DUPLICATE",
    "EXECUTED",
    "INVALID",
    "NOT_OFFERED",
    "REFUSED",
    "UNKNOWN",
    "ToolOutcome",
    "ToolPath",
    "ToolStage",
    "ToolTurn",
]
