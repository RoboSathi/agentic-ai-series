"""Protocols for generic control-loop orchestration."""

from typing import Protocol, runtime_checkable

from agent_platform.control_loop.types import (
    AgentState,
    Checkpoint,
    LoopVerdict,
    StepResult,
    Trajectory,
)


@runtime_checkable
class ReactStep(Protocol):
    """Advances a trajectory by one reason-act-observe step."""

    async def step(self, trajectory: Trajectory) -> StepResult: ...


@runtime_checkable
class LoopController(Protocol):
    """The watchdog contract: return continue, or abort with a reason."""

    def check(self, trajectory: Trajectory) -> LoopVerdict: ...


@runtime_checkable
class StateMachine(Protocol):
    """The legal-transition table for an agent run."""

    def legal_transitions(self, current: AgentState) -> set[AgentState]: ...

    def transition(self, current: AgentState, target: AgentState) -> AgentState:
        """Move states, raising IllegalTransitionError on an illegal move."""
        raise NotImplementedError


@runtime_checkable
class CheckpointStore(Protocol):
    """Persists and restores a run's durable state."""

    def save(self, checkpoint: Checkpoint) -> None: ...

    def restore(self, run_id: str) -> Checkpoint | None: ...
