"""Analytic facts (ADR 0009): one row per unit of work, stamped with where it happened.

Real DuckDB (in memory) behind every store; only the model is a stub.
"""

from collections.abc import Iterator
from typing import Any

import pytest

from agent_platform.config import get_settings
from agent_platform.llm.call_log import RecordedClient
from agent_platform.llm.client import LLMClient
from agent_platform.llm.types import (
    CompletionRequest,
    CompletionResult,
    HealthCheckResult,
    TokenUsage,
)
from agent_platform.logging.trace_context import bound_log_context
from agent_platform.observability.facts import (
    CLOSED,
    OPENED,
    DuckDbFactStore,
    Fact,
    ModelCallFact,
    NullFactStore,
    TaskEventFact,
    TurnFact,
    create_fact_store,
    fact_store,
    fact_table_ddl,
    install_fact_store,
)
from agent_platform.observability.store import DuckDbObservabilityStore

REQUEST = CompletionRequest(messages=[])
USAGE = TokenUsage(prompt_tokens=11, completion_tokens=7, total_tokens=18)


class _Model(LLMClient):
    def __init__(self, name: str, fails: bool = False) -> None:
        self.name, self.fails = name, fails

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        if self.fails:
            raise RuntimeError(f"{self.name} down")
        return CompletionResult(content="ok", model=self.name, finish_reason="stop", usage=USAGE)

    async def health_check(self) -> HealthCheckResult:
        return HealthCheckResult(healthy=True)

    @property
    def model(self) -> str:
        return self.name

    @property
    def tier(self) -> str:
        return "provider"


class _Collector:
    def __init__(self) -> None:
        self.facts: list[Fact] = []

    def append(self, fact: Fact) -> None:
        self.facts.append(fact)

    def open_task(self, agent_id: str, session_id: str) -> tuple[str, str] | None:
        return None


@pytest.fixture
def store() -> DuckDbObservabilityStore:
    return DuckDbObservabilityStore(":memory:")


@pytest.fixture
def installed() -> Iterator[None]:
    previous = fact_store()
    yield
    install_fact_store(previous)


def _rows(store: DuckDbObservabilityStore, table: str) -> list[dict[str, Any]]:
    return list(store.iter_rows(f"SELECT * FROM {table}"))


# -- the store ---------------------------------------------------------------------------


def test_a_fact_type_implies_its_table() -> None:
    ddl = fact_table_ddl(TurnFact)
    assert f"CREATE TABLE IF NOT EXISTS {TurnFact.table}" in ddl
    assert '"latency_ms" DOUBLE' in ddl and '"steps" BIGINT' in ddl
    assert '"recorded_at" TIMESTAMPTZ' in ddl and '"screens" VARCHAR[]' in ddl


def test_a_fact_is_stamped_with_the_context_it_happened_in(
    store: DuckDbObservabilityStore,
) -> None:
    facts = DuckDbFactStore(store, agent_id="fallback-agent")
    with bound_log_context(agent="agent-a", session="s-1", trace_id="t-1", channel="tui"):
        facts.append(ModelCallFact(model="m"))
    (row,) = _rows(store, ModelCallFact.table)
    assert (row["agent_id"], row["session_id"], row["channel"]) == ("agent-a", "s-1", "tui")


def test_outside_a_turn_the_stores_agent_and_default_channel_are_used(
    store: DuckDbObservabilityStore,
) -> None:
    DuckDbFactStore(store, agent_id="fallback-agent").append(ModelCallFact(model="m"))
    (row,) = _rows(store, ModelCallFact.table)
    assert row["agent_id"] == "fallback-agent"
    assert row["channel"] == get_settings().analytics.default_channel
    assert row["turn_id"] == "" and row["step"] is None


def test_an_unobserved_field_is_null_not_zero(store: DuckDbObservabilityStore) -> None:
    DuckDbFactStore(store).append(TurnFact(status="answered"))
    (row,) = _rows(store, TurnFact.table)
    assert row["status"] == "answered" and row["claims_checked"] is None


def test_appending_never_fails_the_caller(store: DuckDbObservabilityStore) -> None:
    store.close()
    DuckDbFactStore(store).append(ModelCallFact(model="m"))  # logged, not raised


def test_a_batch_lands_in_one_table_per_fact_type(store: DuckDbObservabilityStore) -> None:
    DuckDbFactStore(store).append_many(
        [ModelCallFact(model="a"), TurnFact(status="answered"), ModelCallFact(model="b")]
    )
    assert sorted(row["model"] for row in _rows(store, ModelCallFact.table)) == ["a", "b"]
    assert len(_rows(store, TurnFact.table)) == 1


def test_a_table_from_an_earlier_version_gains_the_new_columns(
    store: DuckDbObservabilityStore,
) -> None:
    store.execute_script(f"CREATE TABLE {TurnFact.table} (agent_id VARCHAR, status VARCHAR)")
    store.execute(f"INSERT INTO {TurnFact.table} VALUES ('old-agent', 'answered')")
    DuckDbFactStore(store).append(TurnFact(status="escalated", steps=2))
    rows = sorted(
        (row["status"], row["steps"])
        for row in store.iter_rows(f"SELECT status, steps FROM {TurnFact.table}")
    )
    assert rows == [("answered", None), ("escalated", 2)]


def test_an_open_explicit_task_is_found_until_it_is_closed(
    store: DuckDbObservabilityStore,
) -> None:
    facts = DuckDbFactStore(store, agent_id="agent-a")
    opened = TaskEventFact(
        agent_id="agent-a", session_id="s", task_id="t1", event=OPENED, kind="refund"
    )
    facts.append(opened)
    assert facts.open_task("agent-a", "s") == ("t1", "refund")
    facts.append(opened.model_copy(update={"event": CLOSED}))
    assert facts.open_task("agent-a", "s") is None


# -- which store ----------------------------------------------------------------------------


def test_without_a_database_nothing_is_kept() -> None:
    assert isinstance(create_fact_store(None, "agent"), NullFactStore)


def test_with_a_database_facts_are_persisted(store: DuckDbObservabilityStore) -> None:
    assert isinstance(create_fact_store(store, "agent"), DuckDbFactStore)


@pytest.mark.usefixtures("installed")
def test_the_installed_store_is_the_one_every_component_appends_to() -> None:
    collector = _Collector()
    install_fact_store(collector)
    assert fact_store() is collector


# -- model calls ------------------------------------------------------------------------


async def test_a_call_is_recorded_with_its_tokens_and_role() -> None:
    sink = _Collector()
    await RecordedClient(_Model("primary"), role="planner", position=0, store=sink).complete(
        REQUEST
    )
    (fact,) = sink.facts
    assert isinstance(fact, ModelCallFact)
    assert (fact.role, fact.model, fact.is_backup, fact.total_tokens) == (
        "planner",
        "primary",
        False,
        18,
    )
    assert fact.succeeded and fact.finish_reason == "stop" and fact.latency_ms >= 0


async def test_a_failed_backup_call_is_recorded_and_still_raises() -> None:
    sink = _Collector()
    client = RecordedClient(_Model("m", fails=True), role="r", position=1, store=sink)
    with pytest.raises(RuntimeError):
        await client.complete(REQUEST)
    (fact,) = sink.facts
    assert isinstance(fact, ModelCallFact)
    assert not fact.succeeded and "down" in fact.error and fact.is_backup
