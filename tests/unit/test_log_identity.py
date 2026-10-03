"""One turn, one trace: every log line a turn causes names the agent, session and trace.

The trace id used to be bound only inside a tracer span, and the `@logged`
CALLED/RETURNED lines (most of every log file) never carried it, so one
turn's lines could not be pulled out of eighteen files. Sessions were named
`tui-…`, after the screen, so nothing in a line said which agent wrote it.
"""

import logging
from collections.abc import Iterator
from typing import Any

import pytest

from agent_platform.config import get_settings
from agent_platform.logging import (
    LoggerFactory,
    bound_channel,
    bound_log_context,
    log_context_fields,
    new_session_id,
)
from agent_platform.persistence import SqliteConnectionPool
from customer_support_agent.domain.identity import set_current_customer
from customer_support_agent.graph import build_agent
from tests.conftest import CUSTOMER_CODE, ScriptedLLM

_FIELDS = "_agentic_structured_fields"


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[tuple[str, dict[str, Any]]] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append((record.name, dict(getattr(record, _FIELDS, {}) or {})))


@pytest.fixture
def captured() -> Iterator[_Capture]:
    handler = _Capture()
    LoggerFactory.subscribe(handler)
    try:
        yield handler
    finally:
        LoggerFactory.unsubscribe(handler)


def _traces(records: list[tuple[str, dict[str, Any]]]) -> set[str]:
    return {fields["trace_id"] for _, fields in records if "trace_id" in fields}


class TestEveryLineOfATurnCarriesItsTrace:
    async def test_one_turn_is_one_trace_across_subsystems(
        self, seeded_pool: SqliteConnectionPool, captured: _Capture
    ) -> None:
        set_current_customer(None)
        agent = build_agent(llm=ScriptedLLM(), pool=seeded_pool, with_memory=False)
        with bound_channel("test"):
            await agent.handle(CUSTOMER_CODE, session_id="s1")
        stamped = [(name, f) for name, f in captured.records if "trace_id" in f]
        assert len(_traces(stamped)) == 1
        assert len({name for name, _ in stamped}) > 1, "more than one subsystem's lines"
        assert all(
            f["agent"] == get_settings().agent.id and f["session"] == "s1" for _, f in stamped
        )
        assert all(f["channel"] == "test" for _, f in stamped)

    async def test_each_turn_has_its_own_trace(
        self, seeded_pool: SqliteConnectionPool, captured: _Capture
    ) -> None:
        set_current_customer(None)
        agent = build_agent(llm=ScriptedLLM(), pool=seeded_pool, with_memory=False)
        await agent.handle(CUSTOMER_CODE, session_id="s2")
        first = _traces(captured.records)
        captured.records.clear()
        await agent.handle("show orders", session_id="s2")
        assert first and _traces(captured.records) and first.isdisjoint(_traces(captured.records))

    async def test_nothing_stays_bound_after_the_turn(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        set_current_customer(None)
        agent = build_agent(llm=ScriptedLLM(), pool=seeded_pool, with_memory=False)
        await agent.handle(CUSTOMER_CODE, session_id="s3")
        assert log_context_fields() == {}

    async def test_the_run_record_names_the_trace(self, seeded_pool: SqliteConnectionPool) -> None:
        set_current_customer(None)
        agent = build_agent(llm=ScriptedLLM(), pool=seeded_pool, with_memory=False)
        await agent.handle(CUSTOMER_CODE, session_id="s4")
        [record] = agent.runs.recent(1)
        assert len(record.trace_id) == 32


class TestIdsNameTheirAgent:
    def test_a_session_id_starts_with_the_agent(self) -> None:
        assert new_session_id("customer_support_agent", "-", 8).startswith(
            "customer_support_agent-"
        )

    def test_a_bound_context_restores_on_exit(self) -> None:
        with bound_log_context(agent="a", session="s", trace_id="t", channel="c"):
            assert log_context_fields() == {
                "agent": "a",
                "session": "s",
                "channel": "c",
                "trace_id": "t",
            }
        assert log_context_fields() == {}
