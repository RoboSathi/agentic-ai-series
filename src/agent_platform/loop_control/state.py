"""What a loop knows about itself, in a form a checkpoint can keep.

`LoopState` is the governor's whole memory: the phase, what every counter has spent
(including the seconds a run has already used, so a resumed run cannot start a fresh clock),
every action attempted with its outcome, the evidence each step added, and the last verdict.
It is a plain model: an agent keeps it in its LangGraph state, the checkpointer persists it,
and a resumed governor is restored from it — so a repeat is still a repeat after a restart.

`LoopStateMachine` is the phase table: the legal moves, and an error for any other.
"""

from pydantic import BaseModel, Field

from agent_platform.loop_control.types import (
    IllegalTransitionError,
    LoopPhase,
    LoopVerdict,
    TerminalOutcome,
    Transition,
)

_LEGAL: dict[LoopPhase, frozenset[LoopPhase]] = {
    "IDLE": frozenset({"PLANNING", "DONE", "FAILED"}),
    # A skill's prefetch reads run before the first model call: PLANNING → WAITING_TOOL.
    "PLANNING": frozenset({"ACTING", "WAITING_TOOL", "WAITING_HUMAN", "DONE", "FAILED"}),
    "ACTING": frozenset({"WAITING_TOOL", "PLANNING", "DONE", "FAILED"}),
    "WAITING_TOOL": frozenset({"ACTING", "PLANNING", "DONE", "FAILED"}),
    "WAITING_HUMAN": frozenset({"PLANNING", "DONE", "FAILED"}),
    "DONE": frozenset({"PLANNING"}),
    "FAILED": frozenset({"PLANNING"}),
}


class ActionRecord(BaseModel):
    """One action the model asked for, and what became of it."""

    action: str
    fingerprint: str
    canonical: str
    # What it returned, or why it was refused, failed or not run.
    outcome: str = ""
    succeeded: bool = False
    # Not run: a repeat the governor suppressed.
    suppressed: bool = False


class LoopState(BaseModel):
    """The governor's state: everything a resumed loop must not forget."""

    phase: LoopPhase = "IDLE"
    transitions: list[Transition] = Field(default_factory=list)
    step: int = 0
    # Counter → spent (`steps`, `tool_calls`, `tokens`, `cost_usd`).
    spent: dict[str, float] = Field(default_factory=dict)
    # Wall-clock seconds this run used before the current process took it over.
    seconds: float = 0.0
    actions: list[ActionRecord] = Field(default_factory=list)
    # The new evidence each step's results added, oldest first (convergence).
    gains: list[float] = Field(default_factory=list)
    # What each successful step returned, most recent last (the evidence seen so far).
    evidence: list[str] = Field(default_factory=list)
    verdict: LoopVerdict | None = None
    outcome: TerminalOutcome | None = None


class LoopStateMachine:
    """The legal phase transitions of a loop (`StateMachine`)."""

    def legal_transitions(self, current: LoopPhase) -> set[LoopPhase]:
        return set(_LEGAL[current])

    def transition(self, state: LoopState, target: LoopPhase, reason: str = "") -> LoopState:
        """Move `state` to `target`, or raise `IllegalTransitionError`. Staying put is a no-op."""
        if target == state.phase:
            return state
        if target not in _LEGAL[state.phase]:
            raise IllegalTransitionError(f"{state.phase} → {target} is not a legal transition")
        state.transitions.append(Transition(from_state=state.phase, to_state=target, reason=reason))
        state.phase = target
        return state


__all__ = ["ActionRecord", "LoopState", "LoopStateMachine"]
