"""Context source for generic working memory."""

from agentic_commons.context.types import Chunk, Task
from agentic_commons.memory.working import WorkingMemoryStore


class WorkingMemoryContextSource:
    """Expose bounded session working memory to the common context pipeline."""

    def __init__(self, working_memory: WorkingMemoryStore) -> None:
        self._working_memory = working_memory

    async def candidates(self, task: Task) -> list[Chunk]:
        """Return conversation history and compact task state as context."""
        memory = self._working_memory.get(task.session_id)
        candidates: list[Chunk] = []
        if memory.history:
            history = "\n".join(
                f"{item.get('role')}: {item.get('content', '')}"
                if isinstance(item, dict)
                else str(item)
                for item in memory.history
            )
            candidates.append(
                Chunk(content=history, kind="history", source="working_memory", pinned=True)
            )
        if memory.state:
            state = "\n".join(f"{key}: {value}" for key, value in memory.state.items())
            candidates.append(
                Chunk(content=state, kind="memory", source="working_memory", pinned=True)
            )
        return candidates
