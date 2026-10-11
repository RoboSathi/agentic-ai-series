"""Context source for generic working memory."""

from agent_platform.context.types import TRUST, Chunk, Task
from agent_platform.memory.working import WorkingMemoryStore

WORKING_MEMORY = "working_memory"


class WorkingMemoryContextSource:
    """Expose bounded session working memory to the common context pipeline."""

    def __init__(self, working_memory: WorkingMemoryStore) -> None:
        self._working_memory = working_memory

    async def candidates(self, task: Task) -> list[Chunk]:
        """Return conversation history and compact task state as context.

        The compact state is pinned and the transcript is not, which is
        the whole of the difference between them. State is a handful of
        short lines naming who is speaking and what they are speaking
        about — it anchors the turn, and the window can always afford it.
        The transcript is unbounded in width: one turn that rendered a
        table of every order the customer has ever placed will sit in
        every subsequent window, and pinning it meant no budget, no
        deduplication and no reduction strategy could touch it. Unpinned,
        it competes for space like everything else and is trimmed first
        when the turn cannot afford it.
        """
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
                Chunk(
                    content=history,
                    kind="history",
                    source=WORKING_MEMORY,
                    metadata={TRUST: "user"},
                )
            )
        if memory.state:
            state = "\n".join(f"{key}: {value}" for key, value in memory.state.items())
            candidates.append(
                Chunk(
                    content=state,
                    kind="memory",
                    source=WORKING_MEMORY,
                    pinned=True,
                    # The harness's own bookkeeping (who is verified, what is under discussion).
                    metadata={TRUST: "system"},
                )
            )
        return candidates
