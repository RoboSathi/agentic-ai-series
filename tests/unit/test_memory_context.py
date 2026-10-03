"""Long-term memory as the model reads it.

Recall used to hand the model `event_tui-d8b725e7_ab84…: query: show orders`
three times over: an event id it cannot use, and this session's own
earlier turns, which working memory already carries (with the assistant's
replies too). Recall into context adds only what working memory cannot know.
"""

from typing import Any

from agent_platform.context.types import Task
from agent_platform.memory.context_source import SUBJECT_ID, MemoryRecallContextSource
from agent_platform.memory.types import MemoryFact, Provenance, ScoredFact

_SESSION = "tui-current"


class _Facade:
    def __init__(self, *facts: MemoryFact) -> None:
        self._facts = facts

    async def recall(self, query: Any) -> list[ScoredFact]:
        return [ScoredFact(fact=fact, score=0.5, tier="lexical") for fact in self._facts]


def _event(text: str, session: str, key: str = "") -> MemoryFact:
    return MemoryFact(
        key=key or f"event_{session}_{abs(hash(text + session)):x}",
        value=text,
        memory_type="episodic",
        category="event",
        provenance=Provenance(source="agent"),
        metadata={"session_id": session},
    )


def _semantic(key: str, value: str) -> MemoryFact:
    return MemoryFact(
        key=key, value=value, memory_type="semantic", provenance=Provenance(source="customer")
    )


async def _candidates(*facts: MemoryFact) -> list[Any]:
    task = Task(query="refund", session_id=_SESSION, metadata={SUBJECT_ID: "89"})
    return await MemoryRecallContextSource(_Facade(*facts)).candidates(task)  # type: ignore[arg-type]


class TestAnEventIsItsText:
    async def test_no_event_id_reaches_the_model(self) -> None:
        [chunk] = await _candidates(_event("query: refund for order 10417", "tui-earlier"))
        assert chunk.content == "query: refund for order 10417"
        assert "event_" not in chunk.source and "event_" not in chunk.content

    async def test_a_semantic_fact_keeps_the_key_that_names_it(self) -> None:
        [chunk] = await _candidates(_semantic("preferred_language", "French"))
        assert chunk.content == "preferred_language: French"
        assert chunk.source == "memory:semantic:preferred_language"


class TestWorkingMemoryCoversThisSession:
    async def test_this_sessions_own_events_are_not_recalled(self) -> None:
        chunks = await _candidates(
            _event("query: show orders", _SESSION),
            _event("query: refund for order 10417", "tui-earlier"),
        )
        assert [c.content for c in chunks] == ["query: refund for order 10417"]

    async def test_repeats_of_one_event_collapse(self) -> None:
        chunks = await _candidates(
            _event("query: show orders", "tui-a", key="event_a1"),
            _event("Query: show orders", "tui-b", key="event_b1"),
        )
        assert len(chunks) == 1
