"""The loop's contracts: one verdict, one set of terminal outcomes, one phase table.

Every loop-control mechanism — the step cap, the budgets, the deadline, the stagnation tiers,
convergence, the historical baseline — answers with a `LoopVerdict`. Routers read its
`decision`; cards and logs read the rest. A run ends in exactly one `TerminalOutcome`, mapped
once onto the durable run's status and the tool path's stage, so no agent invents its own.

Runtime logic lives in `governor.py`; state in `state.py`.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from agent_platform.observability.types import utc_now

# Where a loop is in its lifecycle (`state.LoopStateMachine` holds the legal moves).
LoopPhase = Literal["IDLE", "PLANNING", "ACTING", "WAITING_TOOL", "WAITING_HUMAN", "DONE", "FAILED"]
# Which mechanism decided. One name per mechanism, so a verdict always says who spoke.
WatchdogLayer = Literal[
    "step_count",
    "wall_clock",
    "token_budget",
    "cost_budget",
    "tool_budget",
    "duplicate_action",
    "semantic_stagnation",
    "cycle",
    "convergence",
    "historical",
]
# What the loop does next:
#   continue  go round again
#   suppress  this one action is not run (a repeat); the loop goes on
#   finalize  stop acting and answer now, from what there is
#   abort     stop without an answer
#   pause     wait for a person (a human-in-the-loop interrupt)
LoopDecision = Literal["continue", "suppress", "finalize", "abort", "pause"]
CheckStatus = Literal["PASS", "FAIL"]
# How a turn or run ended.
TerminalOutcome = Literal[
    "answered", "finalized_on_cap", "aborted", "cancelled", "escalated", "failed"
]

# Terminal outcome → durable run status (`durability.types.RunStatus`).
RUN_STATUS: dict[TerminalOutcome, str] = {
    "answered": "completed",
    "finalized_on_cap": "completed",
    "escalated": "completed",
    "aborted": "aborted",
    "cancelled": "cancelled",
    "failed": "failed_resumable",
}
# Loop decision about one action → the tool path's stage (`tools.path.ToolStage`).
TOOL_STAGE: dict[LoopDecision, str] = {
    "suppress": "duplicate",
    "abort": "refused",
    "finalize": "refused",
    "pause": "refused",
    "continue": "executed",
}


class IllegalTransitionError(Exception):
    """Raised when a transition is not in the legal-transition table."""


class Transition(BaseModel):
    """One recorded phase change."""

    from_state: LoopPhase
    to_state: LoopPhase
    reason: str = ""
    at: datetime = Field(default_factory=utc_now)


class ControlCheck(BaseModel):
    """One deterministic watchdog check, as it stood when the verdict was given."""

    check_id: str
    name: str
    status: CheckStatus
    reason_code: str
    reason: str
    observed: dict[str, Any] = Field(default_factory=dict)
    evidence: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=utc_now)


class BudgetLine(BaseModel):
    """One counter on a verdict: what is spent of what is allowed."""

    spent: float
    cap: float


class LoopVerdict(BaseModel):
    """The one loop-control decision contract every mechanism and every agent shares."""

    decision: LoopDecision
    # Which mechanism decided; None for a plain continue no check objected to.
    layer: WatchdogLayer | None = None
    reason_code: str = "LOOP_CONTROL_CONTINUE"
    reason: str = ""
    checks: list[ControlCheck] = Field(default_factory=list)
    round_number: int = 0
    max_rounds: int = 0
    fingerprint: str = ""
    previous_result: str = ""
    # Every budget counter as it stood, for the LOOP CONTROL card's budget tree.
    budgets: dict[str, BudgetLine] = Field(default_factory=dict)

    @property
    def stops(self) -> bool:
        """Whether the loop goes no further (it answers now, or not at all)."""
        return self.decision in ("finalize", "abort", "pause")

    @property
    def intervened(self) -> bool:
        """Whether any mechanism changed what would otherwise have happened."""
        return self.decision != "continue"


__all__ = [
    "RUN_STATUS",
    "TOOL_STAGE",
    "BudgetLine",
    "CheckStatus",
    "ControlCheck",
    "IllegalTransitionError",
    "LoopDecision",
    "LoopPhase",
    "LoopVerdict",
    "TerminalOutcome",
    "Transition",
    "WatchdogLayer",
]
