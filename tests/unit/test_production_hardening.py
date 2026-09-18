"""The pieces that make the agent production-grade rather than a demo.

Four gaps closed here:

  1. No customer identity. Any caller could read *and refund* any
     customer's order — an IDOR, and the exact risk E31 maps.
  2. The escalation path dead-ended. Refunds queued for human approval
     and nothing could approve them.
  3. Goodwill credits had no rules at all, while refunds had several.
  4. Six subsystems were built, tested and never wired into the request
     path.
"""

import pytest

from agentic_commons.persistence import SqliteConnectionPool
from customer_support_agent.core import (
    Customer,
    HitlQueueRepository,
    OrdersRepository,
    RefundLedgerRepository,
    RefundSaga,
    ReviewQueueService,
    SupportTools,
    build_agent,
    set_current_customer,
)
from tests.conftest import ScriptedLLM, tool_call


@pytest.fixture
def tools(seeded_pool: SqliteConnectionPool) -> SupportTools:
    orders = OrdersRepository(seeded_pool)
    refunds = RefundLedgerRepository(seeded_pool)
    return SupportTools(
        orders, HitlQueueRepository(seeded_pool), refunds, RefundSaga(orders, refunds)
    )


@pytest.fixture
def review(seeded_pool: SqliteConnectionPool) -> ReviewQueueService:
    orders = OrdersRepository(seeded_pool)
    refunds = RefundLedgerRepository(seeded_pool)
    return ReviewQueueService(HitlQueueRepository(seeded_pool), orders, RefundSaga(orders, refunds))


class TestOrdersAreScopedToTheirOwner:
    def test_a_customer_cannot_read_another_customers_order(self, tools: SupportTools) -> None:
        """Order 10003 belongs to u2; the default fixture is u1."""
        assert tools.get_order_status("10003").success is False

    def test_the_refusal_does_not_confirm_the_order_exists(self, tools: SupportTools) -> None:
        """A distinct "not yours" message is itself a disclosure.

        The two messages differ only by the id the caller supplied, which
        tells them nothing they did not already know.
        """
        missing = (tools.get_order_status("99999").error or "").replace("99999", "ID")
        not_mine = (tools.get_order_status("10003").error or "").replace("10003", "ID")
        assert missing == not_mine

    async def test_a_customer_cannot_refund_another_customers_order(
        self, tools: SupportTools, seeded_pool: SqliteConnectionPool
    ) -> None:
        """The part that costs money."""
        result = await tools.process_refund("10003", 45.0)
        assert result.success is False
        assert OrdersRepository(seeded_pool).get("10003").refund_status == "completed"  # type: ignore[union-attr]

    def test_an_unauthenticated_request_sees_nothing(self, tools: SupportTools) -> None:
        """Scope to nothing, never to everything."""
        set_current_customer(None)
        assert tools.get_order_status("10001").success is False

    def test_the_owner_still_sees_their_own_order(self, tools: SupportTools) -> None:
        assert tools.get_order_status("10001").success is True

    def test_identity_is_not_a_tool_argument(self) -> None:
        """Anything in the schema is something the model can choose."""
        from customer_support_agent.core.tools import GoodwillArgs, OrderLookupArgs

        for schema in (OrderLookupArgs, GoodwillArgs):
            fields = set(schema.model_fields)
            assert not fields & {"customer_id", "user_id"}, schema.__name__


class TestTheEscalationPathDrains:
    async def test_a_queued_refund_can_be_approved(
        self, tools: SupportTools, review: ReviewQueueService, seeded_pool: SqliteConnectionPool
    ) -> None:
        tools.queue_for_approval("10002", 210.0)
        request_id = review.pending()[0].id
        outcome = await review.approve(request_id, reviewer="supervisor-1")  # type: ignore[arg-type]
        assert outcome.refund_issued is True
        assert OrdersRepository(seeded_pool).get("10002").refund_status == "completed"  # type: ignore[union-attr]

    def test_a_queued_refund_can_be_denied_without_paying(
        self, tools: SupportTools, review: ReviewQueueService, seeded_pool: SqliteConnectionPool
    ) -> None:
        tools.queue_for_approval("10002", 210.0)
        outcome = review.deny(review.pending()[0].id, "supervisor-1", "not eligible")  # type: ignore[arg-type]
        assert outcome.decision == "denied"
        assert RefundLedgerRepository(seeded_pool).total_for_user("u1") == 0.0

    async def test_eligibility_is_rechecked_at_approval(
        self, tools: SupportTools, review: ReviewQueueService, seeded_pool: SqliteConnectionPool
    ) -> None:
        """An order can change state while it waits in the queue."""
        tools.queue_for_approval("10002", 210.0)
        OrdersRepository(seeded_pool).set_refund_status("10002", "completed")
        outcome = await review.approve(review.pending()[0].id, "supervisor-1")  # type: ignore[arg-type]
        assert outcome.refund_issued is False

    async def test_approving_a_missing_request_is_not_an_error(
        self, review: ReviewQueueService
    ) -> None:
        assert (await review.approve(999, "supervisor-1")).decision == "not_found"

    def test_deciding_drains_the_queue(
        self, tools: SupportTools, review: ReviewQueueService
    ) -> None:
        tools.queue_for_approval("10002", 210.0)
        review.deny(review.pending()[0].id, "supervisor-1")  # type: ignore[arg-type]
        assert review.pending_count() == 0

    def test_the_agent_has_no_route_to_approval(self, tools: SupportTools) -> None:
        """`approve_hitl_request` is a human-controlled surface."""
        assert "approve_hitl_request" not in {tool.name for tool in tools.definitions()}

    async def test_the_human_decision_is_audited_under_the_reviewer(
        self, tools: SupportTools, seeded_pool: SqliteConnectionPool
    ) -> None:
        """The point of escalation is that a person took responsibility."""
        from agentic_commons.audit import SqliteAuditLedger

        audit = SqliteAuditLedger(seeded_pool)
        orders = OrdersRepository(seeded_pool)
        service = ReviewQueueService(
            HitlQueueRepository(seeded_pool),
            orders,
            RefundSaga(orders, RefundLedgerRepository(seeded_pool)),
            audit=audit,
        )
        tools.queue_for_approval("10002", 210.0)
        await service.approve(service.pending()[0].id, "supervisor-7")  # type: ignore[arg-type]
        assert any(entry.actor == "supervisor-7" for entry in audit.by_decision("allow"))


class TestGoodwillIsBounded:
    def test_a_single_credit_is_capped(self, tools: SupportTools) -> None:
        assert tools.apply_goodwill_credit(500.0).success is False

    def test_the_per_customer_total_is_capped(self, tools: SupportTools) -> None:
        """Otherwise the per-credit cap is bypassed by asking repeatedly."""
        for _ in range(4):
            tools.apply_goodwill_credit(25.0)
        assert tools.apply_goodwill_credit(25.0).success is False

    def test_each_credit_is_recorded_separately(
        self, tools: SupportTools, seeded_pool: SqliteConnectionPool
    ) -> None:
        """Colliding ids would let INSERT OR REPLACE hide one."""
        for _ in range(3):
            tools.apply_goodwill_credit(10.0)
        assert RefundLedgerRepository(seeded_pool).total_for_user("u1") == 30.0

    def test_a_credit_goes_to_the_authenticated_customer(
        self, tools: SupportTools, seeded_pool: SqliteConnectionPool
    ) -> None:
        set_current_customer(Customer(customer_id="u2"))
        tools.apply_goodwill_credit(10.0)
        assert RefundLedgerRepository(seeded_pool).total_for_user("u1") == 0.0


class TestSubsystemsAreActuallyWired:
    async def test_the_model_call_runs_through_the_resilience_stack(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        """The model is a remote dependency like any other."""
        agent = build_agent(
            llm=ScriptedLLM([{"content": "hello"}]), pool=seeded_pool, with_memory=False
        )
        await agent.handle("who are you", customer=Customer(customer_id="u1"))
        assert agent._resilient.healthy is True

    async def test_cost_is_attributed_to_the_request(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        agent = build_agent(
            llm=ScriptedLLM([{"content": "hello"}]), pool=seeded_pool, with_memory=False
        )
        await agent.handle("who are you", session_id="s1", customer=Customer(customer_id="u1"))
        assert agent._costs is not None and agent._costs.total_cost > 0

    async def test_spans_are_recorded_for_the_run(self, seeded_pool: SqliteConnectionPool) -> None:
        agent = build_agent(
            llm=ScriptedLLM([{"content": "hello"}]), pool=seeded_pool, with_memory=False
        )
        await agent.handle("who are you", customer=Customer(customer_id="u1"))
        assert agent._tracer is not None and agent._tracer.spans

    async def test_the_reply_is_checked_against_the_tool_results(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        """A claim no tool supports is recorded, not silently served."""
        llm = ScriptedLLM(
            [
                {"tool_calls": tool_call("support.get_order_status", order_id="10001")},
                {"content": "Order 10001 was delivered. We also sent you a free holiday."},
            ]
        )
        agent = build_agent(llm=llm, pool=seeded_pool, with_memory=False)
        state = await agent.handle(
            "status of order 10001", session_id="s1", customer=Customer(customer_id="u1")
        )
        assert any("grounding" in decision for decision in state.decisions)

    async def test_grounding_is_advisory_not_blocking(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        """A detector that can be wrong must not withhold the answer."""
        llm = ScriptedLLM(
            [
                {"tool_calls": tool_call("support.get_order_status", order_id="10001")},
                {"content": "Order 10001 was delivered. We also sent you a free holiday."},
            ]
        )
        agent = build_agent(llm=llm, pool=seeded_pool, with_memory=False)
        state = await agent.handle(
            "status of order 10001", session_id="s1", customer=Customer(customer_id="u1")
        )
        assert state.response

    async def test_slos_are_recorded(self, seeded_pool: SqliteConnectionPool) -> None:
        agent = build_agent(
            llm=ScriptedLLM([{"content": "hello"}]), pool=seeded_pool, with_memory=False
        )
        await agent.handle("who are you", customer=Customer(customer_id="u1"))
        assert agent._slos is not None
        assert agent._slos.status("tool_failure_rate").sample_count > 0
