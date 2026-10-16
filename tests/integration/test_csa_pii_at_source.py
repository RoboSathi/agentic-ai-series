"""A customer's name is written nowhere while `pii` is on, and shown to them as it is (P-40).

The bug this guards: the turn summary showed `verified_customer: Pythagoras (account #19)`.
A real support turn runs here against the seeded database with a scripted model: the
customer verifies, then asks for their orders. Every log record, every card and the turn
summary, the agent's state, its working memory and every database table but `users` itself
must hold tokens; the reply restored for the customer holds their name. `/harness pii off`
puts the raw name back everywhere, which is what the demo shows.
"""

import logging
from collections.abc import Iterator

import pytest

from agent_platform.config import FeatureSet, get_settings, set_active_features
from agent_platform.core.persistence import SqliteConnectionPool
from agent_platform.logging import LoggerFactory
from agent_platform.observability import InMemoryLiveStatusFeed, StageReporter
from agent_platform.observability.types import StatusEvent
from agent_platform.security.pii.active import restore_if_on
from customer_support_agent.domain.identity import set_current_customer
from customer_support_agent.graph import build_agent
from customer_support_agent.graph.state import AgentState
from tests.conftest import CUSTOMER_CODE, ScriptedLLM

# The seeded customer behind CUSTOMER_CODE (`tests/conftest.py`).
NAME = "Ada Lovelace"
EMAIL = "ada@example.com"
# The table that is the record of who the customer is; everything else is the harness's.
SYSTEM_OF_RECORD = "users"


class _Capture(logging.Handler):
    """Every log record, as the text a file handler would be given."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        fields = getattr(record, "_agentic_structured_fields", {}) or {}
        self.lines.append(f"{record.getMessage()} {fields}")


@pytest.fixture
def captured() -> Iterator[_Capture]:
    handler = _Capture()
    LoggerFactory.subscribe(handler)
    try:
        yield handler
    finally:
        LoggerFactory.unsubscribe(handler)
        set_active_features(None)


def _posture(pii: bool) -> FeatureSet:
    return get_settings().features.with_override("pii", pii)


async def _verified_turns(
    pool: SqliteConnectionPool, pii: bool
) -> tuple[list[AgentState], list[StatusEvent], object]:
    set_current_customer(None)
    feed = InMemoryLiveStatusFeed(max_events=0)
    events: list[StatusEvent] = []
    feed.subscribe(events.append)
    agent = build_agent(
        llm=ScriptedLLM(), pool=pool, features=_posture(pii), reporter=StageReporter(feed, "t")
    )
    states = [
        await agent.handle(CUSTOMER_CODE, session_id="pii"),
        await agent.handle("show my orders", session_id="pii"),
    ]
    return states, events, agent


def _written_tables(pool: SqliteConnectionPool) -> str:
    """Every row of every table but the system of record, as one text."""
    tables = [
        row["name"]
        for row in pool.iter_rows("SELECT name FROM sqlite_master WHERE type = 'table'")
        if row["name"] != SYSTEM_OF_RECORD
    ]
    return " ".join(str(list(pool.iter_rows(f'SELECT * FROM "{table}"'))) for table in tables)


def _everything(
    states: list[AgentState], events: list[StatusEvent], agent: object, logs: _Capture
) -> dict[str, str]:
    held = getattr(agent, "_working_memory").get("pii")
    return {
        "logs": " ".join(logs.lines),
        "cards": " ".join(f"{e.summary} {e.detail} {e.fields}" for e in events),
        "state": " ".join(state.model_dump_json() for state in states),
        "working_memory": f"{held.state} {held.history}",
    }


class TestWithPiiOn:
    async def test_the_name_is_written_nowhere(
        self, seeded_pool: SqliteConnectionPool, captured: _Capture
    ) -> None:
        states, events, agent = await _verified_turns(seeded_pool, pii=True)
        places = {
            **_everything(states, events, agent, captured),
            "db": _written_tables(seeded_pool),
        }
        leaked = [where for where, text in places.items() if NAME in text or EMAIL in text]
        assert leaked == []
        assert any(e.stage == "turn_summary" for e in events)

    async def test_the_customer_reads_their_own_name(
        self, seeded_pool: SqliteConnectionPool, captured: _Capture
    ) -> None:
        states, _, _ = await _verified_turns(seeded_pool, pii=True)
        assert NAME not in states[0].response
        assert NAME in restore_if_on(states[0].response)


class TestWithPiiOff:
    async def test_the_raw_name_is_back_for_the_demo(
        self, seeded_pool: SqliteConnectionPool, captured: _Capture
    ) -> None:
        states, events, agent = await _verified_turns(seeded_pool, pii=False)
        places = _everything(states, events, agent, captured)
        assert NAME in states[0].response
        assert NAME in places["working_memory"]
