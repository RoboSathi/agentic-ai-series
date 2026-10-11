"""The Gatekeeper's standings, its DB-free refund rule, and refund settlement.

Standing is what the principal may do about an action: VOID (nothing — it
had no authority), RATIFIABLE (a person may adopt it), VOIDABLE (it ran,
and may be undone until it settles), VALID (it ran, for good). The refund
rule decides only authority — the amount against the agent's limit — and
touches no database. Settlement is when VOIDABLE stops being undoable.
"""

from datetime import timedelta
from typing import Any

import pytest

from agent_platform.config import get_settings
from agent_platform.gatekeeper import AuthorityContext, AuthorityDecision, GatekeeperImpl
from agent_platform.gatekeeper.ledger import CumulativeLedger
from agent_platform.observability.types import utc_now
from agent_platform.persistence import SqliteConnectionPool
from agent_platform.primitives.money import Money
from agent_platform.security import Authorizer
from agent_platform.security.auth.policy import ActionLimit, AgentPolicy, AgentPolicyRepository
from agent_platform.tools import ToolCall
from customer_support_agent.console.commands import VoidCommand
from customer_support_agent.evals.fixture import policy_seed_sql
from customer_support_agent.gatekeeper import REFUND_ACTION_CLASS, CustomerSupportBusinessRules
from customer_support_agent.persistence import OrdersRepository, RefundLedgerRepository
from customer_support_agent.resilience import RefundSaga, RefundSettlement

_ACTOR = "operator"


def _policy(limit: str | None = "50.00") -> AgentPolicy:
    return AgentPolicy(
        agent_id="agent",
        role="support_agent",
        authority_rank=4,
        permissions={"process_refund", "read_orders"},
        updated_at="2026-01-01T00:00:00+00:00",
        limits=(
            {}
            if limit is None
            else {
                REFUND_ACTION_CLASS: ActionLimit(
                    action_class=REFUND_ACTION_CLASS,
                    per_action=Money.parse(limit),
                    window=Money.parse("4000.00"),
                )
            }
        ),
    )


def _refund(amount: float, voidable: bool = True) -> ToolCall:
    return ToolCall(
        action="support.process_refund",
        params={"order_id": 10001, "amount": amount},
        side_effecting=True,
        voidable=voidable,
        required_permission="process_refund",
    )


class TestStandings:
    def test_each_verdict_has_its_standing(self) -> None:
        def standing(verdict: str) -> str | None:
            return AuthorityDecision(
                verdict=verdict,  # type: ignore[arg-type]
                action="a",
                actor="b",
                reason="",
            ).standing

        assert standing("deny") == "VOID"
        assert standing("escalate") == "RATIFIABLE"
        assert standing("allow") == "VALID"

    @pytest.mark.parametrize(("voidable", "expected"), [(True, "VOIDABLE"), (False, "VALID")])
    async def test_an_allowed_action_is_voidable_only_if_its_tool_says_so(
        self, seeded_pool: SqliteConnectionPool, voidable: bool, expected: str
    ) -> None:
        seeded_pool.execute_script(policy_seed_sql())
        policy = _seeded(seeded_pool)
        gatekeeper = GatekeeperImpl(
            authorizer=Authorizer(policy=policy),
            policy=policy,
            ledger=CumulativeLedger(seeded_pool, enabled=False),
            business_rules=CustomerSupportBusinessRules(policy),
        )
        decision = await gatekeeper.evaluate(
            _refund(32.0, voidable), AuthorityContext(actor=policy.agent_id)
        )
        assert decision.standing == expected

    async def test_a_disabled_ledger_is_not_reported_as_a_check(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        """A control this deployment does not have is absent, not SKIPPED every time."""
        seeded_pool.execute_script(policy_seed_sql())
        policy = _seeded(seeded_pool)
        gatekeeper = GatekeeperImpl(
            authorizer=Authorizer(policy=policy),
            policy=policy,
            ledger=CumulativeLedger(seeded_pool, enabled=False),
        )
        decision = await gatekeeper.evaluate(_refund(32.0), AuthorityContext(actor=policy.agent_id))
        assert all(check.check_id != "gatekeeper.cumulative_limit" for check in decision.checks)


def _seeded(pool: SqliteConnectionPool) -> AgentPolicy:
    from agent_platform.persistence import SqlitePlatformStore

    return AgentPolicyRepository(SqlitePlatformStore(pool, agent_id=get_settings().agent.id)).get()


class TestTheRefundRuleIsAuthorityOnly:
    """It decides on the amount against the limit, and reads no database at all."""

    def _rules(self, limit: str | None = "50.00") -> CustomerSupportBusinessRules:
        return CustomerSupportBusinessRules(
            _policy(limit), money_actions=frozenset({"process_refund"})
        )

    def test_over_the_limit_escalates(self) -> None:
        result = self._rules().evaluate(_refund(137.0), AuthorityContext(actor="a"))
        assert result.verdict == "escalate"
        assert result.reason_code == "EXCEEDS_AGENT_REFUND_LIMIT"

    def test_within_the_limit_passes(self) -> None:
        assert self._rules().evaluate(_refund(32.0), AuthorityContext(actor="a")).verdict == "allow"

    def test_a_read_is_not_its_business(self) -> None:
        read = ToolCall(action="support.get_order_status", params={"order_id": 10001})
        result = self._rules().evaluate(read, AuthorityContext(actor="a"))
        assert result.verdict == "allow"
        assert result.checks[0].status == "SKIPPED"

    def test_no_declared_limit_escalates_nothing(self) -> None:
        result = self._rules(None).evaluate(_refund(9999.0), AuthorityContext(actor="a"))
        assert result.verdict == "allow"


class TestSettlement:
    @pytest.fixture
    def settlement(self, seeded_pool: SqliteConnectionPool) -> RefundSettlement:
        return RefundSettlement(OrdersRepository(seeded_pool), RefundLedgerRepository(seeded_pool))

    async def _issue(self, pool: SqliteConnectionPool, window_seconds: int) -> None:
        orders = OrdersRepository(pool)
        order = orders.get(10001)
        assert order is not None
        saga = RefundSaga(orders, RefundLedgerRepository(pool), window_seconds=window_seconds)
        assert (await saga.issue(order, Money.parse("32.00"), "damaged")).succeeded

    async def test_nothing_settles_before_its_window_closes(
        self, seeded_pool: SqliteConnectionPool, settlement: RefundSettlement
    ) -> None:
        await self._issue(seeded_pool, window_seconds=3600)
        assert await settlement.settle_due(_ACTOR) == []
        assert [order.order_id for order in settlement.issued()] == [10001]

    async def test_a_refund_whose_window_has_closed_settles(
        self, seeded_pool: SqliteConnectionPool, settlement: RefundSettlement
    ) -> None:
        await self._issue(seeded_pool, window_seconds=3600)
        past = (utc_now() - timedelta(seconds=1)).isoformat()
        seeded_pool.execute(
            "UPDATE orders SET refund_settles_at = ? WHERE order_id = 10001", (past,)
        )
        outcomes = await settlement.settle_due(_ACTOR)
        assert [(o.order_id, o.state) for o in outcomes] == [(10001, "settled")]
        assert OrdersRepository(seeded_pool).get(10001).refund_status == "completed"  # type: ignore[union-attr]

    async def test_settling_tells_the_customer_and_only_then(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        told: list[tuple[int, str]] = []

        async def notifier(user_id: int, message: str) -> None:
            told.append((user_id, message))

        settlement = RefundSettlement(
            OrdersRepository(seeded_pool), RefundLedgerRepository(seeded_pool), notifier=notifier
        )
        await self._issue(seeded_pool, window_seconds=3600)
        assert told == []
        await settlement.settle(10001, _ACTOR)
        assert [user for user, _ in told] == [1]

    def test_an_order_with_nothing_issued_cannot_be_voided(
        self, settlement: RefundSettlement
    ) -> None:
        assert settlement.void(10001, _ACTOR).state == "not_issued"


class TestSettlementCommands:
    async def test_void_lists_then_voids(self, seeded_pool: SqliteConnectionPool) -> None:
        orders = OrdersRepository(seeded_pool)
        order = orders.get(10001)
        assert order is not None
        await RefundSaga(orders, RefundLedgerRepository(seeded_pool)).issue(
            order, Money.parse("32.00"), "x"
        )
        command = VoidCommand(seeded_pool)
        listing = await command.run([])
        assert [row.cells[0] for row in listing.rows] == ["10001"]
        assert command.argument_completions([]) == ["10001"]
        voided = await command.run(["10001"])
        assert voided.ok
        assert orders.get(10001).refund_status == "none"  # type: ignore[union-attr]

    async def test_a_settled_refund_can_no_longer_be_voided(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        orders = OrdersRepository(seeded_pool)
        order = orders.get(10001)
        assert order is not None
        await RefundSaga(orders, RefundLedgerRepository(seeded_pool)).issue(
            order, Money.parse("32.00"), "x"
        )
        await RefundSettlement(orders, RefundLedgerRepository(seeded_pool)).settle(10001, _ACTOR)
        assert orders.get(10001).refund_status == "completed"  # type: ignore[union-attr]
        refused: Any = await VoidCommand(seeded_pool).run(["10001"])
        assert refused.ok is False
