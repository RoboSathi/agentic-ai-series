"""The invariants no single module owns (LLD §23).

These are properties of the *assembled* system. Each one is true only
because several subsystems agree, which is exactly why none of them can
be tested from inside one module — and why they are the first things to
break silently during a refactor.
"""

import inspect

from agent_platform.persistence import SqliteConnectionPool
from agent_platform.primitives.money import Money
from customer_support_agent.graph import build_agent
from customer_support_agent.graph.agent import CustomerSupportAgent
from customer_support_agent.persistence import OrdersRepository
from tests.conftest import ScriptedLLM, refunded_to, tool_call


class TestNoExecutionPathBypassesTheGatekeeper:
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


class TestEveryRetryIsIdempotent:
    async def test_a_repeated_refund_pays_once(self, seeded_pool: SqliteConnectionPool) -> None:
        from agent_platform.resilience import SqliteIdempotencyGuard
        from customer_support_agent.persistence import RefundLedgerRepository
        from customer_support_agent.resilience import RefundSaga

        orders = OrdersRepository(seeded_pool)
        refunds = RefundLedgerRepository(seeded_pool)
        saga = RefundSaga(orders, refunds, idempotency=SqliteIdempotencyGuard(seeded_pool))
        order = orders.get(10001)
        assert order is not None
        for _ in range(3):
            await saga.issue(order, 32.0, "damaged")
        assert refunded_to(seeded_pool, 1) == Money.parse("32.00")

    def test_the_retry_policy_cannot_run_unkeyed(self) -> None:
        """Structural, not advisory."""
        from agent_platform.resilience import ExponentialBackoffRetry

        signature = inspect.signature(ExponentialBackoffRetry.execute)
        assert signature.parameters["idempotency_key"].default is inspect.Parameter.empty


class TestEpisodesComposeRatherThanDuplicate:
    def test_the_agent_constructs_no_llm_client_of_its_own(self) -> None:
        """The duplication regression that was found and fixed once already."""
        source = inspect.getsource(CustomerSupportAgent)
        assert "AsyncOpenAI" not in source
        assert "create_llm_client" not in source

    def test_the_agent_uses_the_platform_logger_not_print(self) -> None:
        source = inspect.getsource(CustomerSupportAgent)
        assert "print(" not in source


def _raw_call(name: str, arguments: str):  # type: ignore[no-untyped-def]
    from agent_platform.llm.types import ModelToolCall

    return [ModelToolCall(id=name, name=name, arguments=arguments)]
