"""Refunds apply only to settled orders, and are never volunteered.

Three requirements, from a live review of the agent's replies:

  1. A plain order question must not mention refunds. The agent was
     answering "where is my parcel" with the order's refund status, which
     invites a refund request nobody intended to make.
  2. An order cannot be refunded until it is settled — delivered, or
     cancelled before delivery. An order still in flight has nothing to
     refund yet.
  3. The dummy data must obey the same rule. Generating each column
     independently produced pending orders with completed refunds, a
     state the business cannot reach.
"""

import pytest

from agentic_commons.gatekeeper import AuthorityContext, GatekeeperImpl
from agentic_commons.gatekeeper.ledger import CumulativeLedger
from agentic_commons.persistence import SqliteConnectionPool
from agentic_commons.security import Authorizer
from agentic_commons.tools import ToolCall
from customer_support_agent.core import (
    HitlQueueRepository,
    Order,
    OrdersRepository,
    RefundLedgerRepository,
    RefundSaga,
    SupportTools,
    build_agent,
)
from customer_support_agent.core.intent import IntentClassifier
from customer_support_agent.dummy_data.generate_dummy_data import refund_status_for
from tests.conftest import ScriptedLLM, tool_call


def _order(order_id: str, status: str, refund: str = "none") -> Order:
    return Order(
        order_id=order_id,
        user_id="u1",
        timestamp="2026-02-01",
        order_status=status,  # type: ignore[arg-type]
        refund_status=refund,  # type: ignore[arg-type]
        amount=40.0,
    )


@pytest.fixture
def tools(seeded_pool: SqliteConnectionPool) -> SupportTools:
    orders = OrdersRepository(seeded_pool)
    refunds = RefundLedgerRepository(seeded_pool)
    return SupportTools(
        orders, HitlQueueRepository(seeded_pool), refunds, RefundSaga(orders, refunds)
    )


class TestEligibilityRule:
    @pytest.mark.parametrize("status", ["delivered", "cancelled"])
    def test_a_settled_order_is_refundable(self, status: str) -> None:
        assert _order("1", status).is_refund_eligible is True

    @pytest.mark.parametrize("status", ["pending", "shipped"])
    def test_an_in_flight_order_is_not(self, status: str) -> None:
        assert _order("1", status).is_refund_eligible is False

    def test_the_reason_is_readable_by_a_customer(self) -> None:
        reason = _order("10001", "shipped").refund_ineligibility_reason
        assert "still shipped" in reason and "delivered" in reason

    def test_an_already_refunded_order_says_so(self) -> None:
        reason = _order("10001", "delivered", "completed").refund_ineligibility_reason
        assert "already been refunded" in reason


class TestToolsEnforceIt:
    async def test_an_in_flight_order_cannot_be_refunded(
        self, tools: SupportTools, seeded_pool: SqliteConnectionPool
    ) -> None:
        """Enforced in code, not left to the prompt."""
        OrdersRepository(seeded_pool).save(_order("10050", "shipped"))
        result = await tools.process_refund("10050", 40.0)
        assert result.success is False
        assert "nothing to refund yet" in (result.error or "")

    def test_an_in_flight_order_cannot_be_queued_either(
        self, tools: SupportTools, seeded_pool: SqliteConnectionPool
    ) -> None:
        """Otherwise the rule is bypassed by asking for approval instead."""
        OrdersRepository(seeded_pool).save(_order("10051", "pending"))
        assert tools.queue_for_approval("10051", 40.0).success is False

    async def test_a_cancelled_order_can_be_refunded(
        self, tools: SupportTools, seeded_pool: SqliteConnectionPool
    ) -> None:
        OrdersRepository(seeded_pool).save(_order("10052", "cancelled"))
        assert (await tools.process_refund("10052", 40.0)).success is True


class TestRefundsAreNotVolunteered:
    def test_a_status_lookup_says_nothing_about_refunds(self, tools: SupportTools) -> None:
        """The reported behaviour: every order answer mentioned a refund."""
        output = tools.get_order_status("10001").output
        assert "refund" not in output.casefold()

    def test_a_status_lookup_still_answers_the_question(self, tools: SupportTools) -> None:
        output = tools.get_order_status("10001")
        assert "delivered" in output.output and "32.00" in output.output

    def test_an_order_enquiry_does_not_unlock_the_refund_lookup(self) -> None:
        """Checking refund state is itself a refund conversation."""
        permitted = IntentClassifier().permitted_tools("where is my parcel for order 10001")
        assert permitted == {"get_order_status"}

    def test_a_refund_request_does_unlock_it(self) -> None:
        assert "get_refund_status" in IntentClassifier().permitted_tools("I want a refund")

    async def test_the_agent_does_not_check_refunds_on_a_status_query(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        llm = ScriptedLLM(
            [
                {"tool_calls": tool_call("support.get_refund_status", order_id="10001")},
                {"content": "Order 10001 was delivered."},
            ]
        )
        agent = build_agent(llm=llm, pool=seeded_pool, with_memory=False)
        state = await agent.handle("where is order 10001")
        assert any("not offered" in decision for decision in state.decisions)


class TestOfferingNothingDeniesEverything:
    async def test_an_empty_offer_is_not_an_absent_one(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        """Fail-open trap: [] once meant "no narrowing", so everything passed."""
        gatekeeper = GatekeeperImpl(authorizer=Authorizer(), ledger=CumulativeLedger(seeded_pool))
        decision = await gatekeeper.evaluate(
            ToolCall(action="get_order_status"),
            AuthorityContext(actor="a", roles=["support_agent"], offered_actions=[]),
        )
        assert decision.verdict == "deny" and decision.question == "offered"

    async def test_no_narrowing_still_permits(self, seeded_pool: SqliteConnectionPool) -> None:
        gatekeeper = GatekeeperImpl(authorizer=Authorizer(), ledger=CumulativeLedger(seeded_pool))
        decision = await gatekeeper.evaluate(
            ToolCall(action="get_order_status"),
            AuthorityContext(actor="a", roles=["support_agent"], offered_actions=None),
        )
        assert decision.verdict == "allow"

    async def test_an_unoffered_attempt_is_audited(self, seeded_pool: SqliteConnectionPool) -> None:
        """Reaching for authority you were not given is what audit is for."""
        from agentic_commons.audit import SqliteAuditLedger

        audit = SqliteAuditLedger(seeded_pool)
        gatekeeper = GatekeeperImpl(
            authorizer=Authorizer(), ledger=CumulativeLedger(seeded_pool), audit=audit
        )
        await gatekeeper.evaluate(
            ToolCall(action="process_refund", params={"amount": 10.0}),
            AuthorityContext(
                actor="a", roles=["support_agent"], offered_actions=["get_order_status"]
            ),
        )
        assert audit.by_decision("deny")


class TestDummyDataObeysTheRule:
    @pytest.mark.parametrize("status", ["pending", "shipped"])
    def test_an_in_flight_order_is_generated_with_no_refund(self, status: str) -> None:
        import random

        rng = random.Random(1)
        assert all(refund_status_for(rng, status) == "none" for _ in range(50))

    def test_settled_orders_can_carry_a_refund(self) -> None:
        import random

        rng = random.Random(1)
        produced = {refund_status_for(rng, "delivered") for _ in range(200)}
        assert produced - {"none"}

    def test_the_shipped_database_holds_no_impossible_state(self) -> None:
        """The fixture the agent actually reasons over."""
        pool = SqliteConnectionPool()
        row = pool.query_one(
            "SELECT COUNT(*) AS n FROM orders "
            "WHERE order_status IN ('pending', 'shipped') AND refund_status != 'none'"
        )
        assert row is not None and row["n"] == 0
