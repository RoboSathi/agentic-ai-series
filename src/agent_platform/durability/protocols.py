"""What the console needs from an agent that runs durably."""

from typing import Any, Protocol, runtime_checkable

from agent_platform.durability.registry import RunRegistry
from agent_platform.durability.types import CheckpointInfo, ResumeResult


@runtime_checkable
class DurableAgent(Protocol):
    """An agent whose runs are checkpointed and can be resumed.

    Resuming goes through the agent, not straight to the graph: the agent
    re-binds whatever a run needs that is not graph state — who the
    customer is, a per-turn budget — before the pending node runs again.
    """

    @property
    def runs(self) -> RunRegistry: ...

    async def resume(self, thread_id: str, decision: Any | None = None) -> ResumeResult: ...

    async def abort(self, thread_id: str) -> ResumeResult:
        """End a stopped run for good; its record is kept and it never resumes."""
        ...

    def describe_progress(self, values: dict[str, Any]) -> dict[str, Any]:
        """How far a run got, from its checkpointed values, as the stop report's tree."""
        ...

    def cancel(self) -> bool:
        """Stop the run in progress; False when nothing is running."""
        ...

    async def checkpoint_history(self, thread_id: str) -> list[CheckpointInfo]: ...
