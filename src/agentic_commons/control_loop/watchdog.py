"""Deterministic control-loop watchdogs for agent trajectories.

The control loop owns *whether the agent may continue*, not whether an
individual action is authorized. Gatekeeper and Executor answer the latter;
this package prevents trajectory-level pathologies such as repeated actions
and unbounded tool rounds.

Only watchdogs that are actually wired are evaluated. Future layers such as
wall-clock, token, cost, and semantic-stagnation budgets remain explicit in
the contract but are not reported as passing until an implementation exists.
"""

from datetime import datetime
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from agentic_commons.logging import get_logger
from agentic_commons.observability.types import utc_now
from agentic_commons.tools.types import ToolCall

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


class DuplicateActionDetector:
    """Exact-match duplicate detector for one agent trajectory."""

    def __init__(self) -> None:
        self._seen: dict[str, str] = {}
        self._logger = get_logger("control_loop")

    def seen_before(self, call: ToolCall) -> str | None:
        """Return the previous result for the exact same call, if present."""
        return self._seen.get(action_fingerprint(call))

    def record(self, call: ToolCall, output: str) -> None:
        """Remember a successfully completed call."""
        self._seen[action_fingerprint(call)] = output

    def reset(self) -> None:
        """Clear the detector for a new run."""
        self._seen.clear()

    @property
    def call_count(self) -> int:
        """Number of distinct successful calls recorded in this run."""
        return len(self._seen)

    def reject(self, call: ToolCall) -> str:
        """Explain the suppression and provide the previous result."""
        previous = self.seen_before(call) or ""
        self._logger.warning(
            "Suppressed a repeated tool call",
            action=call.action,
            fingerprint=action_fingerprint(call),
        )
        return (
            f"You already called this and it returned: {previous} "
            "Use that result rather than calling it again."
        )


def action_fingerprint(call: ToolCall) -> str:
    """Stable identity for an exact same action + argument set."""
    arguments = ",".join(f"{key}={call.params[key]}" for key in sorted(call.params))
    return f"{call.action}({arguments})"


class ToolControlLoop:
    """Deterministic trajectory boundary used immediately before execution."""

    def __init__(self, duplicate_detector: DuplicateActionDetector | None = None) -> None:
        self._duplicates = duplicate_detector or DuplicateActionDetector()
        self._logger = get_logger("control_loop")

    def evaluate_round(self, round_number: int, max_rounds: int) -> ControlLoopReport:
        """Check whether another tool round is legal."""
        report = self._round_report(round_number, max_rounds)
        self._log(report)
        return report

    def _round_report(self, round_number: int, max_rounds: int) -> ControlLoopReport:
        passed = round_number <= max_rounds
        check = ControlCheck(
            check_id="control_loop.max_tool_iterations",
            name="Round limit",
            status="PASS" if passed else "FAIL",
            reason_code="ROUND_WITHIN_LIMIT" if passed else "MAX_TOOL_ITERATIONS",
            reason=(
                f"Round {round_number} is within the configured limit of {max_rounds}."
                if passed
                else f"Round {round_number} would exceed the configured limit of {max_rounds}."
            ),
            observed={"current_round": round_number, "max_rounds": max_rounds},
        )
        return ControlLoopReport(
            decision="continue" if passed else "abort",
            reason_code="CONTROL_LOOP_CONTINUE" if passed else "MAX_TOOL_ITERATIONS",
            reason=check.reason,
            checks=[check],
            round_number=round_number,
            max_rounds=max_rounds,
        )

    def before_action(
        self,
        call: ToolCall,
        *,
        round_number: int,
        max_rounds: int,
    ) -> ControlLoopReport:
        """Evaluate active loop guards before one tool action."""
        round_report = self._round_report(round_number, max_rounds)
        if round_report.decision == "abort":
            report = round_report.model_copy()
            report.fingerprint = action_fingerprint(call)
            self._log(report)
            return report

        fingerprint = action_fingerprint(call)
        previous = self._duplicates.seen_before(call)
        duplicate_passed = previous is None
        duplicate_check = ControlCheck(
            check_id="control_loop.duplicate_action",
            name="Duplicate action",
            status="PASS" if duplicate_passed else "FAIL",
            reason_code="NO_DUPLICATE" if duplicate_passed else "DUPLICATE_ACTION",
            reason=(
                "No previous identical action exists in this trajectory."
                if duplicate_passed
                else "The exact action and arguments already succeeded earlier in this trajectory."
            ),
            observed={"action": call.action, "fingerprint": fingerprint},
            evidence={"previous_result": previous or ""},
        )
        report = ControlLoopReport(
            decision="continue" if duplicate_passed else "abort",
            reason_code="CONTROL_LOOP_CONTINUE" if duplicate_passed else "DUPLICATE_ACTION",
            reason=duplicate_check.reason,
            checks=[*round_report.checks, duplicate_check],
            round_number=round_number,
            max_rounds=max_rounds,
            fingerprint=fingerprint,
            previous_result=previous or "",
        )
        self._log(report)
        return report

    def record_success(self, call: ToolCall, output: str) -> None:
        """Remember a successful call for this trajectory."""
        self._duplicates.record(call, output)

    def reject_message(self, call: ToolCall) -> str:
        """Return the previous result to the model when a duplicate is suppressed."""
        return self._duplicates.reject(call)

    def _log(self, report: ControlLoopReport) -> None:
        self._logger.info(
            "Control loop evaluated",
            decision=report.decision,
            reason_code=report.reason_code,
            reason=report.reason,
            round_number=report.round_number,
            max_rounds=report.max_rounds,
            fingerprint=report.fingerprint,
            previous_result=report.previous_result,
            checks=[check.model_dump(mode="json") for check in report.checks],
        )


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
    "ControlCheck",
    "ControlLoopReport",
    "DuplicateActionDetector",
    "IllegalTransitionError",
    "LoopController",
    "LoopDecision",
    "LoopVerdict",
    "ReactStep",
    "StateMachine",
    "StepResult",
    "ToolControlLoop",
    "Trajectory",
    "Transition",
    "WatchdogLayer",
    "action_fingerprint",
]
