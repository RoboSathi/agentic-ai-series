"""Loop control and FSM discipline — PARKED (LLD §15).

Interface-only this phase; the contracts below are normative and the
implementations are the next phase's work.

One package, not two, and that is a design decision rather than a
naming one: the FSM discipline (E23) and the watchdog stack (E20/E05)
govern the same object. An FSM transition *is* what a loop step does, so
splitting them would put "is this loop allowed to keep running?" and "is
it in a legal state?" in two places that must agree and eventually will
not.

`customer_support_agent` currently wires LangGraph's checkpointer
directly from its own thin `control_loop/` folder. That is an explicit,
tracked interim step — it migrates to call this package once this phase
opens, rather than being quietly left as the permanent answer.
"""

from datetime import datetime
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from agentic_commons.observability.types import utc_now

AgentState = Literal[
    "IDLE", "PLANNING", "ACTING", "WAITING_TOOL", "WAITING_HUMAN", "DONE", "FAILED"
]

# The six watchdog layers, in the order they are evaluated. Order is part
# of the contract: the *first* layer to trip is the reported abort reason,
# so a loop that blew its budget while also repeating itself reports the
# budget — the thing an operator can act on.
WatchdogLayer = Literal[
    "step_count",
    "wall_clock",
    "token_budget",
    "cost_budget",
    "duplicate_action",
    "semantic_stagnation",
]

LoopDecision = Literal["continue", "abort"]


class IllegalTransitionError(Exception):
    """Raised when a transition is not in the legal-transition table."""


class Transition(BaseModel):
    """One recorded state change."""

    from_state: AgentState
    to_state: AgentState
    reason: str = ""
    at: datetime = Field(default_factory=utc_now)


class Trajectory(BaseModel):
    """The running record of one agent run."""

    run_id: str
    state: AgentState = "IDLE"
    steps: int = 0
    started_at: datetime = Field(default_factory=utc_now)
    tokens_used: int = 0
    cost_usd: float = 0.0
    actions: list[str] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)


class StepResult(BaseModel):
    """The outcome of one loop step."""

    trajectory: Trajectory
    finished: bool = False
    output: str = ""


class LoopVerdict(BaseModel):
    """Whether the loop may continue, and which layer stopped it."""

    decision: LoopDecision = "continue"
    layer: WatchdogLayer | None = None
    reason: str = ""


class Checkpoint(BaseModel):
    """Durable state for resuming a run."""

    run_id: str
    state: AgentState
    transition_history: list[Transition] = Field(default_factory=list)
    durable_facts: dict[str, Any] = Field(default_factory=dict)
    saved_at: datetime = Field(default_factory=utc_now)


@runtime_checkable
class ReactStep(Protocol):
    """Advances a trajectory by one reason-act-observe step."""

    async def step(self, trajectory: Trajectory) -> StepResult: ...


@runtime_checkable
class LoopController(Protocol):
    """The six-layer watchdog stack. Returns continue, or abort with a reason."""

    def check(self, trajectory: Trajectory) -> LoopVerdict: ...


@runtime_checkable
class StateMachine(Protocol):
    """The legal-transition table for an agent run."""

    def legal_transitions(self, current: AgentState) -> set[AgentState]: ...

    def transition(self, current: AgentState, target: AgentState) -> AgentState:
        """Move states, raising IllegalTransitionError on an illegal move."""
        ...


@runtime_checkable
class CheckpointStore(Protocol):
    """Persists and restores a run's durable state."""

    def save(self, checkpoint: Checkpoint) -> None: ...

    def restore(self, run_id: str) -> Checkpoint | None: ...


__all__ = [
    "AgentState",
    "Checkpoint",
    "CheckpointStore",
    "IllegalTransitionError",
    "LoopController",
    "LoopDecision",
    "LoopVerdict",
    "ReactStep",
    "StateMachine",
    "StepResult",
    "Trajectory",
    "Transition",
    "WatchdogLayer",
]
