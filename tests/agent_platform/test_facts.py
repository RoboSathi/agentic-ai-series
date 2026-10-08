"""Analytic facts: one row per model call, tool call and turn, stamped with the turn's identity.

Real DuckDB (in memory) behind every ledger; only the model is a stub.
"""

from typing import Any

import pytest

from agent_platform.config.settings import ModelChainSettings, ModelRef
from agent_platform.executor import ToolExecutorImpl
from agent_platform.executor.types import ApprovedToolCall
from agent_platform.gatekeeper.types import AuthorityContext, AuthorityDecision
from agent_platform.llm.call_log import RecordedClient
from agent_platform.llm.chain import ModelChainBuilder
from agent_platform.llm.client import LLMClient
from agent_platform.llm.types import (
    CompletionRequest,
    CompletionResult,
    HealthCheckResult,
    TokenUsage,
)
from agent_platform.logging.trace_context import bound_log_context
from agent_platform.observability.facts import (
    DuckDbFactLedger,
    Fact,
    LlmCallFact,
    NullFactRecorder,
    ToolCallFact,
    TurnOutcomeFact,
    create_fact_recorder,
    fact_table_ddl,
)
from agent_platform.observability.store import DuckDbObservabilityStore
from agent_platform.tools.types import ToolCall
from customer_support_agent.observability.accounting import TurnAccountant, TurnOutcome

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

    def record(self, fact: Fact) -> None:
        self.facts.append(fact)


class _Handler:
    def __init__(self, fails: bool = False) -> None:
        self.fails = fails

    async def execute(self, call: ToolCall) -> Any:
        if self.fails:
            raise RuntimeError("tool down")
        return {"ok": True}


@pytest.fixture
def store() -> DuckDbObservabilityStore:
    return DuckDbObservabilityStore(":memory:")


def _rows(store: DuckDbObservabilityStore, table: str) -> list[dict[str, Any]]:
    return list(store.iter_rows(f"SELECT * FROM {table}"))


# -- the ledger --------------------------------------------------------------------------


def test_a_fact_type_implies_its_table() -> None:
    ddl = fact_table_ddl(TurnOutcomeFact)
    assert "platform_turn_outcomes" in ddl
    assert "answered BOOLEAN" in ddl and "latency_ms DOUBLE" in ddl
    assert "recorded_at TIMESTAMPTZ" in ddl


def test_a_fact_is_stamped_with_the_turn_it_happened_in(store: DuckDbObservabilityStore) -> None:
    ledger = DuckDbFactLedger(store, agent_id="fallback-agent")
    with bound_log_context(agent="agent-a", session="s-1", trace_id="t-1", channel="tui"):
        ledger.record(LlmCallFact(model="m"))
    (row,) = _rows(store, "platform_llm_calls")
    assert (row["agent_id"], row["session_id"], row["trace_id"], row["channel"]) == (
        "agent-a",
        "s-1",
        "t-1",
        "tui",
    )


def test_outside_a_turn_the_ledgers_agent_is_used(store: DuckDbObservabilityStore) -> None:
    DuckDbFactLedger(store, agent_id="fallback-agent").record(LlmCallFact(model="m"))
    (row,) = _rows(store, "platform_llm_calls")
    assert row["agent_id"] == "fallback-agent" and row["trace_id"] == ""


def test_an_unobserved_field_is_null_not_zero(store: DuckDbObservabilityStore) -> None:
    DuckDbFactLedger(store).record(TurnOutcomeFact(outcome="answered", answered=True))
    (row,) = _rows(store, "platform_turn_outcomes")
    assert row["answered"] is True and row["tool_calls"] is None


def test_recording_never_fails_the_caller(store: DuckDbObservabilityStore) -> None:
    store.close()
    DuckDbFactLedger(store).record(LlmCallFact(model="m"))  # logged, not raised


def test_without_a_store_nothing_is_recorded() -> None:
    assert isinstance(create_fact_recorder(None, "agent"), NullFactRecorder)


def test_with_a_store_the_ledger_is_built(store: DuckDbObservabilityStore) -> None:
    assert isinstance(create_fact_recorder(store, "agent"), DuckDbFactLedger)


# -- model calls -------------------------------------------------------------------------


async def test_a_call_is_recorded_with_its_tokens_and_role() -> None:
    sink = _Collector()
    await RecordedClient(_Model("primary"), sink, role="planner", position=0).complete(REQUEST)
    (fact,) = sink.facts
    assert isinstance(fact, LlmCallFact)
    assert (fact.role, fact.model, fact.position, fact.total_tokens) == (
        "planner",
        "primary",
        0,
        18,
    )
    assert fact.succeeded and fact.finish_reason == "stop" and fact.latency_ms >= 0


async def test_a_failed_call_is_recorded_and_still_raises() -> None:
    sink = _Collector()
    with pytest.raises(RuntimeError):
        await RecordedClient(_Model("m", fails=True), sink, role="r", position=1).complete(REQUEST)
    (fact,) = sink.facts
    assert isinstance(fact, LlmCallFact)
    assert not fact.succeeded and "down" in fact.error and fact.position == 1


async def test_a_chain_records_each_model_it_tried_in_order() -> None:
    sink = _Collector()
    models = {"a": _Model("a", fails=True), "b": _Model("b")}
    builder = ModelChainBuilder(lambda provider, model: models[model], recorder=sink)
    chain = ModelChainSettings(fallbacks=[ModelRef(provider="nvidia", model="b")])
    client = builder.build(ModelRef(provider="nvidia", model="a"), chain, label="answerer")
    await client.complete(REQUEST)
    calls = [fact for fact in sink.facts if isinstance(fact, LlmCallFact)]
    assert [(f.model, f.position, f.succeeded) for f in calls][-1] == ("b", 1, True)
    assert {f.role for f in calls} == {"answerer"}
    assert any(f.model == "a" and not f.succeeded for f in calls)


async def test_a_chain_without_a_recorder_records_nothing() -> None:
    builder = ModelChainBuilder(lambda provider, model: _Model(model))
    client = builder.build(ModelRef(provider="nvidia", model="a"), ModelChainSettings(), "r")
    assert not isinstance(client, RecordedClient)
    assert (await client.complete(REQUEST)).content == "ok"


# -- tool calls --------------------------------------------------------------------------


def _approved(action: str = "support.get_order_status") -> ApprovedToolCall:
    call = ToolCall(action=action)
    decision = AuthorityDecision(verdict="allow", action=action, actor="agent", reason="ok")
    return ApprovedToolCall(call=call, decision=decision, authority=AuthorityContext(actor="agent"))


@pytest.mark.parametrize("fails", [False, True])
async def test_every_execution_is_recorded_with_its_status(fails: bool) -> None:
    sink = _Collector()
    executor = ToolExecutorImpl(handler=_Handler(fails=fails), recorder=sink)
    await executor.execute(_approved())
    (fact,) = sink.facts
    assert isinstance(fact, ToolCallFact)
    assert fact.action == "support.get_order_status" and fact.actor == "agent"
    assert fact.succeeded is not fails
    assert fact.status == ("failed" if fails else "completed")


# -- turns -------------------------------------------------------------------------------


def test_a_settled_turn_is_recorded_as_its_outcome() -> None:
    sink = _Collector()
    accountant = TurnAccountant(facts=sink)
    accountant.begin()
    accountant.settle(TurnOutcome(session_id="s", escalated=True, tool_calls=2, tool_failures=1))
    (fact,) = sink.facts
    assert isinstance(fact, TurnOutcomeFact)
    assert fact.outcome == "escalated" and fact.escalated and not fact.answered
    assert (fact.tool_calls, fact.tool_failures, fact.session_id) == (2, 1, "s")
    assert fact.latency_ms >= 0


def test_a_table_from_an_earlier_version_gains_the_new_columns(
    store: DuckDbObservabilityStore,
) -> None:
    store.execute_script("CREATE TABLE platform_turn_outcomes (agent_id VARCHAR, outcome VARCHAR)")
    store.execute("INSERT INTO platform_turn_outcomes VALUES ('old-agent', 'answered')")
    DuckDbFactLedger(store).record(TurnOutcomeFact(outcome="escalated", unverified_claims=2))
    rows = sorted(
        (row["outcome"], row["unverified_claims"])
        for row in store.iter_rows("SELECT outcome, unverified_claims FROM platform_turn_outcomes")
    )
    assert rows == [("answered", None), ("escalated", 2)]
