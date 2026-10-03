"""Generic data contracts for deterministic agent control loops.

This module contains only contracts and state models. Runtime watchdog logic
lives in :mod:`agent_platform.control_loop.watchdogs` and protocols live in
:mod:`agent_platform.control_loop.protocols`.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from agent_platform.observability.types import utc_now

AgentState = Literal[
    "IDLE", "PLANNING", "ACTING", "WAITING_TOOL", "WAITING_HUMAN", "DONE", "FAILED"
]
WatchdogLayer = Literal[
    "step_count",
    "wall_clock",
    "token_budget",
    "cost_budget",
    "duplicate_action",
    "semantic_stagnation",
]
LoopDecision = Literal["continue", "abort"]
CheckStatus = Literal["PASS", "FAIL"]


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


class ControlCheck(BaseModel):
    """One active deterministic watchdog check."""

    check_id: str
    name: str
    status: CheckStatus
    reason_code: str
    reason: str
    observed: dict[str, Any] = Field(default_factory=dict)
    evidence: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=utc_now)


class ControlLoopReport(BaseModel):
    """Canonical, renderer-friendly report for one control-loop decision."""

    decision: LoopDecision
    reason_code: str = "CONTROL_LOOP_CONTINUE"
    reason: str = ""
    checks: list[ControlCheck] = Field(default_factory=list)
    round_number: int = 0
    max_rounds: int = 0
    fingerprint: str = ""
    previous_result: str = ""
