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

    async def checkpoint_history(self, thread_id: str) -> list[CheckpointInfo]: ...
