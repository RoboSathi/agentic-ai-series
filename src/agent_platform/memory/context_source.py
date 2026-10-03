"""Recalled memory as context candidates.

Adapts `MemoryFacade.recall` to the context pipeline's `CandidateSource`.
Recall is scoped to the task's subject and to nothing wider: a task that
names no subject recalls nothing personal, rather than everyone's.

What reaches the model is what it can use. An episodic event is its text,
not the random id it is stored under. And this session's own events are
left out: working memory already carries the session's conversation, with
the assistant's replies, so recalling them only repeats it. Long-term
recall adds what working memory cannot know — earlier sessions.
"""

from agent_platform.config import get_settings
from agent_platform.context.types import Chunk, Task
from agent_platform.memory.facade import MemoryFacade
from agent_platform.memory.types import MemoryFact, MemoryQuery, ScoredFact

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
        elsewhere = [item for item in recalled if not _this_session(item.fact, task.session_id)]
        return _distinct([_to_chunk(item) for item in elsewhere])


def _this_session(fact: MemoryFact, session_id: str) -> bool:
    """Whether an episodic event was recorded in the session being answered."""
    return bool(session_id) and fact.metadata.get("session_id") == session_id


def _distinct(chunks: list[Chunk]) -> list[Chunk]:
    """The chunks, each text once (case-folded), in order."""
    seen: set[str] = set()
    kept = []
    for chunk in chunks:
        key = chunk.content.casefold()
        if key not in seen:
            seen.add(key)
            kept.append(chunk)
    return kept


def _to_chunk(item: ScoredFact) -> Chunk:
    fact = item.fact
    episodic = fact.memory_type == "episodic"
    return Chunk(
        # A semantic fact is named by its key ("preferred_language: French").
        # An event's key is a random id the model cannot use.
        content=fact.value if episodic else f"{fact.key}: {fact.value}",
        kind="memory",
        source=f"{SOURCE_PREFIX}episodic"
        if episodic
        else f"{SOURCE_PREFIX}{fact.memory_type}:{fact.key}",
        score=item.score,
        metadata={
            "tier": item.tier,
            "provenance": fact.provenance.kind,
            "confirmed": fact.provenance.confirmed,
            "version": fact.version,
        },
    )
