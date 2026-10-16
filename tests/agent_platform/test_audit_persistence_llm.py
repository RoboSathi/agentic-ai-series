"""Audit trail, persistence and the LLM adapters (LLD §3, §4, §5)."""

from types import SimpleNamespace
from typing import Any

import pytest

from agent_platform.audit import AuditEntry, NullAuditLedger, SqliteAuditLedger
from agent_platform.config import get_settings
from agent_platform.core.persistence import IN_MEMORY, ExecutionResult, SqliteConnectionPool
from agent_platform.llm.cascade import CascadingLLMClient, ConfiguredEscalationPolicy
from agent_platform.llm.client import HostedClient, create_llm_client, create_pinned_llm_client
from agent_platform.llm.embedding_cache import CachingEmbeddingClient
from agent_platform.llm.embeddings import (
    HashingEmbeddingClient,
    HostedEmbeddingClient,
    cosine_similarity,
    create_embedding_client,
    tokenize,
)
from agent_platform.llm.metering import metered
from agent_platform.llm.types import CompletionRequest, CompletionResult, Message, TokenUsage


class TestAuditLedger:
    def test_records_and_reconstructs_a_trace(self, pool: SqliteConnectionPool) -> None:
        ledger = SqliteAuditLedger(pool)
        ledger.record(AuditEntry(trace_id="t1", actor="a", decision="allow", action="x"))
        assert len(ledger.reconstruct("t1")) == 1

    def test_reconstruction_is_in_causal_order(self, pool: SqliteConnectionPool) -> None:
        """Insert sequence, not wall clock — two decisions can share a millisecond."""
        ledger = SqliteAuditLedger(pool)
        for action in ("first", "second", "third"):
            ledger.record(AuditEntry(trace_id="t1", actor="a", decision="allow", action=action))
        assert [entry.action for entry in ledger.reconstruct("t1")] == [
            "first",
            "second",
            "third",
        ]

    def test_versions_default_to_the_configured_ones(self, pool: SqliteConnectionPool) -> None:
        ledger = SqliteAuditLedger(pool)
        ledger.record(AuditEntry(trace_id="t1", actor="a", decision="allow"))
        assert ledger.reconstruct("t1")[0].policy_version == "refund-policy-v1"

    def test_the_authority_context_round_trips(self, pool: SqliteConnectionPool) -> None:
        ledger = SqliteAuditLedger(pool)
        ledger.record(
            AuditEntry(
                trace_id="t1",
                actor="a",
                decision="deny",
                authority_context={"granted_by": "untrusted_content", "granted_by_rank": 8},
            )
        )
        assert ledger.reconstruct("t1")[0].authority_context["granted_by_rank"] == 8

    def test_filters_by_decision(self, pool: SqliteConnectionPool) -> None:
        ledger = SqliteAuditLedger(pool)
        ledger.record(AuditEntry(trace_id="t", actor="a", decision="allow"))
        ledger.record(AuditEntry(trace_id="t", actor="a", decision="escalate"))
        assert len(ledger.by_decision("escalate")) == 1

    def test_there_is_no_update_or_delete_method(self) -> None:
        """Append-only enforced by construction: nothing to call, nothing to review."""
        forbidden = {"update", "delete", "edit", "remove", "amend"}
        assert forbidden.isdisjoint(dir(SqliteAuditLedger))

    def test_the_null_ledger_records_nothing(self) -> None:
        ledger = NullAuditLedger()
        ledger.record(AuditEntry(trace_id="t", actor="a", decision="allow"))
        assert ledger.reconstruct("t") == []


class TestPersistence:
    def test_in_memory_state_survives_across_calls(self) -> None:
        """A fresh connection per call would silently get an empty database."""
        pool = SqliteConnectionPool(IN_MEMORY)
        pool.execute("CREATE TABLE t (id INTEGER)")
        pool.execute("INSERT INTO t VALUES (1)")
        assert pool.query_one("SELECT COUNT(*) AS n FROM t") == {"n": 1}
        pool.close()

    def test_query_one_returns_none_when_empty(self, pool: SqliteConnectionPool) -> None:
        pool.execute("CREATE TABLE t (id INTEGER)")
        assert pool.query_one("SELECT * FROM t") is None

    def test_execute_many_inserts_a_batch(self, pool: SqliteConnectionPool) -> None:
        pool.execute("CREATE TABLE t (id INTEGER)")
        pool.execute_many("INSERT INTO t VALUES (?)", [(1,), (2,), (3,)])
        assert pool.query_one("SELECT COUNT(*) AS n FROM t") == {"n": 3}

    def test_iter_rows_streams_results(self, pool: SqliteConnectionPool) -> None:
        pool.execute("CREATE TABLE t (id INTEGER)")
        pool.execute_many("INSERT INTO t VALUES (?)", [(1,), (2,)])
        assert len(list(pool.iter_rows("SELECT * FROM t"))) == 2

    def test_a_script_runs_multiple_statements(self, pool: SqliteConnectionPool) -> None:
        pool.execute_script("CREATE TABLE a (x INT); CREATE TABLE b (y INT);")
        assert pool.query_one("SELECT COUNT(*) AS n FROM sqlite_master") == {"n": 2}

    def test_foreign_keys_are_enforced(self, pool: SqliteConnectionPool) -> None:
        pool.execute_script(
            "CREATE TABLE parent (id INTEGER PRIMARY KEY);"
            "CREATE TABLE child (id INTEGER, parent_id INTEGER REFERENCES parent(id));"
        )
        import sqlite3

        with pytest.raises(sqlite3.IntegrityError):
            pool.execute("INSERT INTO child VALUES (1, 99)")

    def test_execution_result_defaults_are_empty(self) -> None:
        assert ExecutionResult().rows == []


class TestEmbeddings:
    async def test_the_offline_client_is_deterministic(self) -> None:
        client = HashingEmbeddingClient()
        first = await client.embed(["refund"])
        second = await client.embed(["refund"])
        assert first == second

    async def test_related_text_scores_higher_than_unrelated(self) -> None:
        client = HashingEmbeddingClient()
        vectors = await client.embed(["refund money back", "refund money", "weather forecast"])
        assert cosine_similarity(vectors[0], vectors[1]) > cosine_similarity(vectors[0], vectors[2])

    def test_cosine_of_an_empty_vector_is_zero(self) -> None:
        assert cosine_similarity([], []) == 0.0

    def test_tokenize_lowercases_and_splits(self) -> None:
        assert tokenize("Refund, please!") == ["refund", "please"]

    def test_a_client_is_always_available(self) -> None:
        assert create_embedding_client() is not None

    def test_a_hosted_provider_without_its_key_is_never_asked(self) -> None:
        """A request it would refuse still spends a rate-limit slot: go local from the start."""
        llm = get_settings().llm
        hosted = llm.model_copy(
            update={"embedding": llm.embedding.model_copy(update={"backend": "hosted"})}
        )
        assert llm.provider_settings(hosted.embedding.provider).key_env_vars
        assert isinstance(create_embedding_client(hosted), HashingEmbeddingClient)

    def test_a_hosted_provider_with_its_key_is_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        llm = get_settings().llm
        hosted = llm.model_copy(
            update={"embedding": llm.embedding.model_copy(update={"backend": "hosted"})}
        )
        key_env = llm.provider_settings(hosted.embedding.provider).key_env_vars[0]
        monkeypatch.setenv(key_env, "test-key")
        client = create_embedding_client(hosted)
        assert isinstance(client, CachingEmbeddingClient)
        assert isinstance(client.inner, HostedEmbeddingClient)


class TestLLMClient:
    def test_the_factory_returns_a_configured_client(self) -> None:
        assert create_llm_client() is not None

    def test_cost_is_computed_from_configured_rates(self) -> None:
        """Never a hard-coded price."""
        settings = get_settings().llm
        client = create_pinned_llm_client(settings.provider, settings.provider_settings().model)
        assert isinstance(client, HostedClient)
        usage = TokenUsage(prompt_tokens=1000, completion_tokens=1000, total_tokens=2000)
        assert client.cost_for(usage) == pytest.approx(0.0008)

    async def test_the_scripted_client_records_its_requests(self, scripted_llm) -> None:
        await scripted_llm.complete(
            CompletionRequest(messages=[Message(role="user", content="hi")])
        )
        assert len(scripted_llm.requests) == 1


class _FakeCompletions:
    """The SDK's `chat.completions`, answering with a fixed usage."""

    def __init__(self, total_tokens: int) -> None:
        self._total = total_tokens

    async def create(self, **_: Any) -> SimpleNamespace:
        message = SimpleNamespace(content="ok", tool_calls=None, model_extra={})
        return SimpleNamespace(
            model="fake",
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=SimpleNamespace(
                prompt_tokens=self._total, completion_tokens=0, total_tokens=self._total
            ),
        )


def _metered_client(total_tokens: int) -> HostedClient:
    sdk: Any = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions(total_tokens)))
    return HostedClient(client=sdk)


class TestTokenMetering:
    """The adapter meters every call; no agent counts tokens itself."""

    async def test_each_call_reaches_the_runs_sink(self) -> None:
        counted: list[TokenUsage] = []
        client = _metered_client(40)
        request = CompletionRequest(messages=[Message(role="user", content="hi")])
        with metered(counted.append):
            await client.complete(request)
            await client.complete(request)
        assert [usage.total_tokens for usage in counted] == [40, 40]

    async def test_a_call_outside_a_run_is_counted_by_no_one(self) -> None:
        counted: list[int] = []
        request = CompletionRequest(messages=[Message(role="user", content="hi")])
        with metered(counted.append):
            pass
        await _metered_client(40).complete(request)
        assert counted == []


class TestCascade:
    @staticmethod
    def _settings():  # type: ignore[no-untyped-def]
        from agent_platform.config import get_settings

        return get_settings().llm.cascade

    async def test_a_satisfactory_answer_does_not_escalate(self, scripted_llm) -> None:
        cascade = CascadingLLMClient([scripted_llm], self._settings())
        result = await cascade.complete(
            CompletionRequest(messages=[Message(role="user", content="hi")])
        )
        assert result.tier == "scripted"

    async def test_an_empty_answer_escalates_to_the_next_tier(self, scripted_llm) -> None:
        """Cheapest sufficient tier first — 'sufficient' is the operative word."""
        cheap = type(scripted_llm)([{"content": ""}])
        expensive = type(scripted_llm)([{"content": "a real answer"}])
        cascade = CascadingLLMClient([cheap, expensive], self._settings())
        result = await cascade.complete(
            CompletionRequest(messages=[Message(role="user", content="hi")])
        )
        assert result.content == "a real answer"

    def test_a_cascade_needs_at_least_one_tier(self) -> None:
        with pytest.raises(ValueError):
            CascadingLLMClient([], self._settings())

    def test_the_policy_escalates_on_empty_content(self) -> None:
        policy = ConfiguredEscalationPolicy(self._settings())
        assert policy.should_escalate(CompletionResult(content="")) is True

    def test_the_policy_does_not_escalate_a_good_answer(self) -> None:
        policy = ConfiguredEscalationPolicy(self._settings())
        assert policy.should_escalate(CompletionResult(content="answer")) is False

    def test_a_tool_call_alone_counts_as_an_answer(self) -> None:
        """An empty content field with a tool call is not an empty answer."""
        from agent_platform.llm.types import ModelToolCall

        policy = ConfiguredEscalationPolicy(self._settings())
        result = CompletionResult(content="", tool_calls=[ModelToolCall(name="x")])
        assert policy.should_escalate(result) is False
