"""Recalled memory as context candidates.

Adapts `MemoryFacade.recall` to the context pipeline's `CandidateSource`.
Recall is scoped to the task's subject and to nothing wider: a task that
names no subject recalls nothing personal, rather than everyone's.
"""

from agentic_commons.config import get_settings
from agentic_commons.context.types import Chunk, Task
from agentic_commons.memory.facade import MemoryFacade
from agentic_commons.memory.types import MemoryQuery, ScoredFact

# The key in `Task.metadata` naming whose memory may be recalled.
SUBJECT_ID = "subject_id"
# Prefix on a chunk's `source`, so a trace shows it came from memory.
SOURCE_PREFIX = "memory:"


class MemoryRecallContextSource:
    """Contributes the subject's most relevant long-term memories."""

    def __init__(self, facade: MemoryFacade, limit: int | None = None) -> None:
        self._facade = facade
        self._limit = limit or get_settings().context.top_k

    async def candidates(self, task: Task) -> list[Chunk]:
        subject = str(task.metadata.get(SUBJECT_ID) or "")
        if not subject or not task.query.strip():
            return []
        recalled = await self._facade.recall(
            MemoryQuery(text=task.query, subject_id=subject, limit=self._limit)
        )
        return [_to_chunk(item) for item in recalled]


def _to_chunk(item: ScoredFact) -> Chunk:
    fact = item.fact
    return Chunk(
        content=f"{fact.key}: {fact.value}",
        kind="memory",
        source=f"{SOURCE_PREFIX}{fact.memory_type}:{fact.key}",
        score=item.score,
        metadata={
            "tier": item.tier,
            "provenance": fact.provenance.kind,
            "confirmed": fact.provenance.confirmed,
            "version": fact.version,
        },
    )
