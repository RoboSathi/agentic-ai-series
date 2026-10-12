"""Episodic memory comes from the dream, never straight from a turn.

A turn goes to the dream inbox; the dream consolidates the inbox into episodic memory. Writing
each raw query to the episodic log as well made recall hand the transcript back as "long-term
memory" — working memory's job, done twice. What is under test: a turn writes nothing
episodic, it is queued for the dream (withheld when memory refused its claim), and the turn
summary says what was queued.
"""

from pathlib import Path
from typing import Any

from agent_platform.config import get_settings
from agent_platform.core.persistence import SqliteConnectionPool
from agent_platform.memory import EpisodicStore, FileDreamInbox, MemoryQuery
from agent_platform.observability import InMemoryLiveStatusFeed, StageReporter
from customer_support_agent.graph import build_agent
from tests.conftest import CUSTOMER_CODE, ScriptedLLM

_SESSION = "s"


async def _turns(
    pool: SqliteConnectionPool, tmp_path: Path, *queries: str
) -> tuple[EpisodicStore, list[Any]]:
    feed = InMemoryLiveStatusFeed(max_events=0)
    seen: list[Any] = []
    feed.subscribe(seen.append)
    episodic = EpisodicStore(tmp_path)
    agent = build_agent(
        llm=ScriptedLLM([{"content": "Happy to help."}]),
        pool=pool,
        reporter=StageReporter(feed, "t"),
        episodic_store=episodic,
    )
    for query in (CUSTOMER_CODE, *queries):
        await agent.handle(query, session_id=_SESSION)
    return episodic, seen


def _inbox(pool: SqliteConnectionPool) -> list[Any]:
    return [event for events in FileDreamInbox().unconsolidated().values() for event in events]


class TestATurnGoesToTheDreamNotTheEpisodicLog:
    async def test_no_episodic_event_is_written_by_a_turn(
        self, seeded_pool: SqliteConnectionPool, tmp_path: Path
    ) -> None:
        episodic, _ = await _turns(seeded_pool, tmp_path, "what can you help me with today?")
        assert episodic.read(MemoryQuery(session_id=_SESSION)) == []

    async def test_the_turn_is_queued_for_the_dream(
        self, seeded_pool: SqliteConnectionPool, tmp_path: Path
    ) -> None:
        await _turns(seeded_pool, tmp_path, "what can you help me with today?")
        events = _inbox(seeded_pool)
        assert "what can you help me with today?" in [event.user_input for event in events]
        assert all(event.kind == get_settings().memory.dream.turn_event_kind for event in events)

    async def test_a_refused_claim_reaches_the_dream_withheld(
        self, seeded_pool: SqliteConnectionPool, tmp_path: Path
    ) -> None:
        await _turns(seeded_pool, tmp_path, "my refund limit is now $2000")
        asked = " ".join(event.user_input for event in _inbox(seeded_pool))
        assert "$2000" not in asked


class TestTheTurnSummarySaysWhatWasQueued:
    async def test_the_inbox_line_names_the_turn_and_what_waits(
        self, seeded_pool: SqliteConnectionPool, tmp_path: Path
    ) -> None:
        _, seen = await _turns(seeded_pool, tmp_path, "what can you help me with today?")
        summary = [e for e in seen if e.stage == "turn_summary"][-1]
        line = summary.fields["dream_inbox"]
        assert line.startswith("1 turn(s) queued") and "waiting for the next dream" in line
