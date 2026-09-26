"""The invariants no single module owns (LLD §23).

These are properties of the *assembled* system. Each one is true only
because several subsystems agree, which is exactly why none of them can
be tested from inside one module — and why they are the first things to
break silently during a refactor.
"""

import inspect

from agentic_commons.audit import SqliteAuditLedger
from agentic_commons.gatekeeper import AuthorityContext, GatekeeperImpl
from agentic_commons.gatekeeper.ledger import CumulativeLedger
from agentic_commons.memory import (
    EpisodicStore,
    MemoryStore,
    ProceduralStore,
    SemanticStore,
    WorkingMemory,
)
from agentic_commons.money import Money
from agentic_commons.persistence import SqliteConnectionPool
from agentic_commons.security import Authorizer
from agentic_commons.tools import ToolCall
from customer_support_agent.graph import build_agent
from customer_support_agent.graph.agent import CustomerSupportAgent
from customer_support_agent.persistence import OrdersRepository
from tests.conftest import ScriptedLLM, tool_call


class TestNoExecutionPathBypassesTheGatekeeper:
    async def test_every_tool_call_is_evaluated(self, seeded_pool: SqliteConnectionPool) -> None:
        """One call, one decision — no silent side door."""
        audit = SqliteAuditLedger(seeded_pool)
        gatekeeper = GatekeeperImpl(
            authorizer=Authorizer(), ledger=CumulativeLedger(seeded_pool), audit=audit
        )
        llm = ScriptedLLM(
            [
                {"tool_calls": tool_call("support.get_order_status", order_id=10001)},
                {"tool_calls": tool_call("support.get_refund_status", order_id=10001)},
                {"content": "Here's the status."},
            ]
        )
        agent = build_agent(llm=llm, pool=seeded_pool, gatekeeper=gatekeeper, with_memory=False)
        await agent.handle("status of order 10001")
        assert audit.count() == 2

    def test_the_graph_has_exactly_one_tool_invocation_path(self) -> None:
        """A second call site is how a bypass gets introduced by accident."""
        source = inspect.getsource(CustomerSupportAgent)
        assert source.count("registry.invoke(") == 1

    def test_that_path_evaluates_before_it_invokes(self) -> None:
        source = inspect.getsource(CustomerSupportAgent._invoke)
        assert source.index("gatekeeper.evaluate") < source.index("registry.invoke")

    async def test_a_denied_call_never_reaches_the_tool(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        llm = ScriptedLLM(
            [
                {"tool_calls": tool_call("support.process_refund", order_id=10002, amount=210.0)},
                {"content": "Queued instead."},
            ]
        )
        agent = build_agent(llm=llm, pool=seeded_pool, with_memory=False)
        await agent.handle("refund order 10002")
        assert OrdersRepository(seeded_pool).get(10002).refund_status == "none"  # type: ignore[union-attr]


class TestNoMalformedOutputReachesAnExecutor:
    async def test_an_unparseable_tool_call_is_not_executed(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        """Coercing a malformed call is how an agent acts on a value no model produced."""
        llm = ScriptedLLM(
            [
                {"tool_calls": _raw_call("support.process_refund", "{not json at all")},
                {"content": "I need valid arguments."},
            ]
        )
        agent = build_agent(llm=llm, pool=seeded_pool, with_memory=False)
        state = await agent.handle("refund something")
        assert OrdersRepository(seeded_pool).get(10001).refund_status == "none"  # type: ignore[union-attr]
        assert any("unavailable" in result for result in state.tool_results)

    async def test_missing_required_arguments_are_rejected(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        llm = ScriptedLLM(
            [
                {"tool_calls": tool_call("support.process_refund", order_id=10001)},
                {"content": "I need the amount."},
            ]
        )
        agent = build_agent(llm=llm, pool=seeded_pool, with_memory=False)
        await agent.handle("refund 10001")
        assert OrdersRepository(seeded_pool).get(10001).refund_status == "none"  # type: ignore[union-attr]


class TestEverySideEffectHasATrace:
    async def test_a_refund_produces_an_audit_entry_with_its_versions(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        """Attributable to a versioned (model, policy, data, authority) tuple."""
        audit = SqliteAuditLedger(seeded_pool)
        gatekeeper = GatekeeperImpl(
            authorizer=Authorizer(), ledger=CumulativeLedger(seeded_pool), audit=audit
        )
        await gatekeeper.evaluate(
            ToolCall(action="process_refund", params={"amount": 20.0}),
            AuthorityContext(actor="agent-1", roles=["support_agent"], granted_by="user_task"),
        )
        entry = audit.by_decision("allow")[0]
        assert entry.policy_version and entry.data_version and entry.model_version

    async def test_a_denial_is_audited_as_loudly_as_an_approval(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        audit = SqliteAuditLedger(seeded_pool)
        gatekeeper = GatekeeperImpl(
            authorizer=Authorizer(), ledger=CumulativeLedger(seeded_pool), audit=audit
        )
        await gatekeeper.evaluate(
            ToolCall(action="update_refund_policy"),
            AuthorityContext(actor="agent-1", roles=["support_agent"]),
        )
        assert audit.by_decision("deny")

    def test_pii_is_redacted_before_a_trace_persists(
        self, seeded_pool: SqliteConnectionPool, security_on: None
    ) -> None:
        """Read from storage, not from the live view — persistence is the claim.

        The in-process span keeps its raw text so the TUI can show the
        operator what actually happened; what lands on disk, and outlives
        the request, must not.
        """
        from agentic_commons.observability import LocalTracer
        from agentic_commons.security import PIIFilterImpl

        tracer = LocalTracer(seeded_pool, redactor=PIIFilterImpl())
        with tracer.span("lookup", result="customer ada@example.com asked") as span:
            trace_id = span.trace_id
        stored = seeded_pool.query_one(
            "SELECT result FROM trace_spans WHERE trace_id = ?", (trace_id,)
        )
        assert stored is not None
        assert "ada@example.com" not in stored["result"]
        assert "[REDACTED]" in stored["result"]


class TestEveryRetryIsIdempotent:
    async def test_a_repeated_refund_pays_once(self, seeded_pool: SqliteConnectionPool) -> None:
        from agentic_commons.resilience import SqliteIdempotencyGuard
        from customer_support_agent.persistence import RefundLedgerRepository
        from customer_support_agent.resilience import RefundSaga

        orders = OrdersRepository(seeded_pool)
        refunds = RefundLedgerRepository(seeded_pool)
        saga = RefundSaga(orders, refunds, idempotency=SqliteIdempotencyGuard(seeded_pool))
        order = orders.get(10001)
        assert order is not None
        for _ in range(3):
            await saga.issue(order, 32.0, "damaged")
        assert refunds.total_for_user(1) == Money.parse("32.00")

    def test_the_retry_policy_cannot_run_unkeyed(self) -> None:
        """Structural, not advisory."""
        from agentic_commons.resilience import ExponentialBackoffRetry

        signature = inspect.signature(ExponentialBackoffRetry.execute)
        assert signature.parameters["idempotency_key"].default is inspect.Parameter.empty


class TestProtocolsHaveRealImplementations:
    def test_every_memory_store_actually_implements_the_protocol(
        self, pool: SqliteConnectionPool, tmp_path
    ) -> None:
        """A declared-but-unimplemented Protocol was a caught regression."""
        stores = [
            WorkingMemory(),
            EpisodicStore(tmp_path),
            SemanticStore(pool),
            ProceduralStore(pool),
        ]
        for store in stores:
            assert isinstance(store, MemoryStore), type(store).__name__

    def test_no_memory_store_is_a_stub(self, pool: SqliteConnectionPool, tmp_path) -> None:
        """Every CoALA class is first-class and real, never stubbed."""
        for store in (WorkingMemory(), EpisodicStore(tmp_path), SemanticStore(pool)):
            body = inspect.getsource(type(store).write)
            assert "pass" not in body.split("\n")[-2:]


class TestEpisodesComposeRatherThanDuplicate:
    def test_the_agent_constructs_no_llm_client_of_its_own(self) -> None:
        """The duplication regression that was found and fixed once already."""
        source = inspect.getsource(CustomerSupportAgent)
        assert "AsyncOpenAI" not in source
        assert "create_llm_client" not in source

    def test_the_agent_runs_no_auth_check_of_its_own(self) -> None:
        source = inspect.getsource(CustomerSupportAgent)
        assert "roles" in source  # it passes them along...
        assert "permissions_for" not in source  # ...but never evaluates them

    def test_the_agent_uses_the_platform_logger_not_print(self) -> None:
        source = inspect.getsource(CustomerSupportAgent)
        assert "print(" not in source


def _raw_call(name: str, arguments: str):  # type: ignore[no-untyped-def]
    from agentic_commons.llm.types import ModelToolCall

    return [ModelToolCall(id=name, name=name, arguments=arguments)]
