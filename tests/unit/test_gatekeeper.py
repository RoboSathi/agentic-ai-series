"""The Gatekeeper's four questions (LLD §11).

The truth table below is the heart of the suite: every combination of
surface, authority rank and cumulative state, asserted rather than
assumed. The four E12 live attacks live in the integration tests, where
they run against the whole assembled harness.
"""

import asyncio
from typing import Any

import pytest

from agentic_commons.audit import NullAuditLedger, SqliteAuditLedger
from agentic_commons.config import get_settings
from agentic_commons.executor import ApprovedToolCall
from agentic_commons.gatekeeper import (
    AuthorityContext,
    AuthorityHierarchyResolver,
    GatekeeperImpl,
)
from agentic_commons.gatekeeper.ledger import CumulativeLedger
from agentic_commons.gatekeeper.mandate import MandateSet
from agentic_commons.gatekeeper.tier2 import NullTier2Advisor
from agentic_commons.gatekeeper.types import LedgerEntry
from agentic_commons.money import Money
from agentic_commons.persistence import SqliteConnectionPool
from agentic_commons.security import Authorizer
from agentic_commons.tools import ToolCall
from customer_support_agent.domain.order_reference import ORDER_UNDER_DISCUSSION
from customer_support_agent.gatekeeper.business_rules import CustomerSupportBusinessRules
from customer_support_agent.gatekeeper.mandates import EstablishedOrderScope
from customer_support_agent.graph import build_agent
from customer_support_agent.persistence.repositories import OrdersRepository
from tests.conftest import ScriptedLLM, tool_call


@pytest.fixture
def gatekeeper(seeded_pool: SqliteConnectionPool, agent_policy: Any) -> GatekeeperImpl:
    return GatekeeperImpl(
        authorizer=Authorizer(policy=agent_policy),
        policy=agent_policy,
        ledger=CumulativeLedger(seeded_pool),
        audit=SqliteAuditLedger(seeded_pool),
    )


@pytest.fixture
def agent_authority() -> AuthorityContext:
    return AuthorityContext(actor="agent-1", roles=["support_agent"], granted_by="user_task")


class TestAuthorityHierarchy:
    def test_ranks_follow_the_configured_order(self) -> None:
        resolver = AuthorityHierarchyResolver()
        assert resolver.rank_of("provider") == 1
        assert resolver.rank_of("user_task") == 5
        assert resolver.rank_of("untrusted_content") == 8

    @pytest.mark.parametrize(
        "source,expected",
        [
            ("provider", True),
            ("organization", True),
            ("user_task", True),
            ("runtime_reminders", False),
            ("tool_observations", False),
            ("untrusted_content", False),
        ],
    )
    def test_only_the_top_ranks_may_grant(self, source: str, expected: bool) -> None:
        assert AuthorityHierarchyResolver().can_grant(source) is expected

    def test_an_unknown_source_is_treated_as_the_weakest(self) -> None:
        """Fail-safe defaults: unplaced means untrusted, not middling."""
        resolver = AuthorityHierarchyResolver()
        assert resolver.can_grant("something_invented") is False

    def test_a_caller_cannot_assert_its_own_rank(self) -> None:
        """Otherwise every component would claim rank 1."""
        resolver = AuthorityHierarchyResolver()
        claimed = AuthorityContext(actor="a", granted_by="untrusted_content", granted_by_rank=1)
        assert resolver.resolve(claimed).granted_by_rank == 8

    def test_outranks_compares_two_sources(self) -> None:
        resolver = AuthorityHierarchyResolver()
        assert resolver.outranks("provider", "user_task") is True
        assert resolver.outranks("tool_observations", "user_task") is False


class TestCumulativeLedger:
    def test_maps_an_action_to_its_budget_class(self, pool: SqliteConnectionPool) -> None:
        assert CumulativeLedger(pool).class_of("support.process_refund") == "refund"

    def test_accumulates_within_the_window(self, pool: SqliteConnectionPool) -> None:
        ledger = CumulativeLedger(pool)
        for _ in range(3):
            ledger.record(
                LedgerEntry(ledger_key="a", action_class="refund", actor="a", amount=100.0)
            )
        assert ledger.window_total("a", "refund") == Money.parse("300.00")

    def test_detects_a_breach_before_it_happens(self, pool: SqliteConnectionPool) -> None:
        """The check asks 'would this exceed?', not 'did it?'."""
        ledger = CumulativeLedger(pool)
        ledger.record(LedgerEntry(ledger_key="a", action_class="refund", actor="a", amount=3990.0))
        assert ledger.would_exceed("a", "refund", Money.parse("5.00")) is False
        assert ledger.would_exceed("a", "refund", Money.parse("50.00")) is True

    def test_budgets_are_per_key(self, pool: SqliteConnectionPool) -> None:
        ledger = CumulativeLedger(pool)
        ledger.record(LedgerEntry(ledger_key="a", action_class="refund", actor="a", amount=4000.0))
        assert ledger.would_exceed("b", "refund", Money.parse("100.00")) is False

    def test_an_unconfigured_class_is_uncapped(self, pool: SqliteConnectionPool) -> None:
        assert CumulativeLedger(pool).threshold_for("invented") is None


class TestGatekeeperDecisions:
    async def test_permitted_action_is_allowed(
        self, gatekeeper: GatekeeperImpl, agent_authority: AuthorityContext
    ) -> None:
        decision = await gatekeeper.evaluate(ToolCall(action="get_order_status"), agent_authority)
        assert decision.verdict == "allow"

    async def test_locked_surface_is_denied(
        self, gatekeeper: GatekeeperImpl, agent_authority: AuthorityContext
    ) -> None:
        decision = await gatekeeper.evaluate(
            ToolCall(action="update_refund_policy"), agent_authority
        )
        assert decision.verdict == "deny"
        assert decision.question == "surface"

    async def test_human_controlled_surface_escalates(
        self, gatekeeper: GatekeeperImpl, agent_authority: AuthorityContext
    ) -> None:
        decision = await gatekeeper.evaluate(
            ToolCall(action="approve_hitl_request"), agent_authority
        )
        assert decision.verdict == "escalate"

    async def test_an_actor_without_the_role_is_denied(self, gatekeeper: GatekeeperImpl) -> None:
        viewer = AuthorityContext(actor="v", roles=["viewer"], granted_by="user_task")
        decision = await gatekeeper.evaluate(
            ToolCall(action="process_refund", params={"amount": 10.0}), viewer
        )
        assert decision.verdict == "deny"
        assert decision.question == "actor"

    @pytest.mark.parametrize(
        "source", ["runtime_reminders", "tool_observations", "untrusted_content"]
    )
    async def test_low_rank_authority_cannot_act(
        self, gatekeeper: GatekeeperImpl, source: str
    ) -> None:
        """No phrasing in untrusted content can raise its own rank."""
        context = AuthorityContext(actor="agent-1", roles=["support_agent"], granted_by=source)
        decision = await gatekeeper.evaluate(
            ToolCall(action="process_refund", params={"amount": 10.0}), context
        )
        assert decision.verdict == "deny"
        assert decision.question == "authority"

    async def test_amount_over_the_ceiling_escalates(
        self, gatekeeper: GatekeeperImpl, agent_authority: AuthorityContext
    ) -> None:
        decision = await gatekeeper.evaluate(
            ToolCall(action="process_refund", params={"amount": 500.0}), agent_authority
        )
        assert decision.verdict == "escalate"
        assert decision.question == "automation"

    async def test_every_decision_is_audited(
        self, pool: SqliteConnectionPool, agent_authority: AuthorityContext
    ) -> None:
        """One decision, exactly one audit entry — no silent paths."""
        audit = SqliteAuditLedger(pool)
        gatekeeper = GatekeeperImpl(
            authorizer=Authorizer(), ledger=CumulativeLedger(pool), audit=audit
        )
        await gatekeeper.evaluate(ToolCall(action="get_order_status"), agent_authority)
        await gatekeeper.evaluate(ToolCall(action="update_refund_policy"), agent_authority)
        assert audit.count() == 2

    async def test_tier2_cannot_change_the_verdict(
        self, pool: SqliteConnectionPool, agent_authority: AuthorityContext
    ) -> None:
        """A Tier-2 advisor has no field that stops anything."""

        class AlarmedAdvisor(NullTier2Advisor):
            async def flag(self, call: ToolCall):  # type: ignore[no-untyped-def]
                from agentic_commons.gatekeeper.types import AdvisoryFlag

                return AdvisoryFlag(flagged=True, reason="this looks terrible", confidence=1.0)

        gatekeeper = GatekeeperImpl(
            authorizer=Authorizer(), ledger=CumulativeLedger(pool), tier2=AlarmedAdvisor()
        )
        decision = await gatekeeper.evaluate(ToolCall(action="get_order_status"), agent_authority)
        assert decision.verdict == "allow"
        assert decision.advisory is not None and decision.advisory.flagged is True

    async def test_commit_only_records_spent_amounts(
        self, pool: SqliteConnectionPool, agent_authority: AuthorityContext
    ) -> None:
        ledger = CumulativeLedger(pool)
        gatekeeper = GatekeeperImpl(authorizer=Authorizer(), ledger=ledger)
        call = ToolCall(action="process_refund", params={"amount": 25.0})
        await gatekeeper.evaluate(call, agent_authority)
        assert ledger.window_total("agent-1", "refund") == Money.zero()
        gatekeeper.commit(call, agent_authority)
        assert ledger.window_total("agent-1", "refund") == Money.parse("25.00")


class TestWithholdingTheGatekeeper:
    """What an episode is demonstrating when it switches the feature off.

    The settled rule has two halves and both matter: authorization,
    budgets and surface policy stop happening, *and* the audit entry is
    still written. An ungoverned execution that leaves no trace is worse
    than the one the demo is about — nobody watching could tell it
    apart from an execution that never occurred.
    """

    def _agent(self, on: bool, pool: SqliteConnectionPool) -> object:
        settings = get_settings()
        settings.features = settings.features.with_override("gatekeeper", on)
        return build_agent(llm=ScriptedLLM(), pool=pool)

    def test_it_is_wired_when_the_feature_is_declared(
        self, seeded_pool: SqliteConnectionPool, restore_settings: None
    ) -> None:
        assert self._agent(True, seeded_pool)._gatekeeper is not None

    def test_it_is_absent_when_the_feature_is_withheld(
        self, seeded_pool: SqliteConnectionPool, restore_settings: None
    ) -> None:
        assert self._agent(False, seeded_pool)._gatekeeper is None

    def test_the_audit_ledger_survives_the_bypass(
        self, seeded_pool: SqliteConnectionPool, restore_settings: None
    ) -> None:
        """The trail is what still distinguishes it from nothing happening."""
        agent = self._agent(False, seeded_pool)
        assert not isinstance(agent._audit, NullAuditLedger)

    def test_an_ungoverned_approval_says_so_rather_than_mimicking_one(self) -> None:
        """Whatever reads this later must be able to tell the two apart."""
        approved = ApprovedToolCall.ungoverned(
            ToolCall(action="process_refund", params={"amount": 4000.0}),
            AuthorityContext(actor="agent", granted_by="user_task"),
        )
        assert approved.decision.reason_code == "GATEKEEPER_DISABLED"
        assert approved.decision.question == "ungoverned"
        assert "without authorization" in approved.decision.reason


class TestOwnershipIsAnAuthorityQuestion:
    """Another customer's order must be refused here, not further down.

    It used to be refused by arithmetic: the Gatekeeper's business rules
    ran only for `process_refund` and `queue_for_approval`, so a *read*
    of somebody else's order was ALLOWED, executed, and then returned
    nothing because `OrdersRepository.get_for_customer` scopes its SQL.
    The audit trail recorded an authorized read of an order the caller
    had no standing to name, and one forgotten `WHERE user_id` anywhere
    downstream turns that into a disclosure.
    """

    def _rules(self, pool: SqliteConnectionPool, policy: Any) -> CustomerSupportBusinessRules:
        return CustomerSupportBusinessRules(OrdersRepository(pool), None, policy)

    def _verdict(self, pool: SqliteConnectionPool, policy: Any, action: str, order_id: int) -> Any:
        return self._rules(pool, policy).evaluate(
            ToolCall(action=action, params={"order_id": order_id, "amount": 10.0}),
            AuthorityContext(
                actor="agent", granted_by="user_task", request_context={"customer_id": 1}
            ),
        )

    @pytest.mark.parametrize(
        "action", ["get_order_status", "get_refund_status", "process_refund"]
    )
    def test_another_customers_order_is_refused_for_every_action(
        self, action: str, seeded_pool: SqliteConnectionPool, agent_policy: Any
    ) -> None:
        """Order 10003 belongs to customer 2; customer 1 may not touch it."""
        result = self._verdict(seeded_pool, agent_policy, action, 10003)
        assert result.verdict != "allow"
        assert result.reason_code == "ORDER_NOT_FOUND_FOR_CUSTOMER"

    def test_a_nonexistent_order_is_refused_identically(
        self, seeded_pool: SqliteConnectionPool, agent_policy: Any
    ) -> None:
        """Same refusal either way, so the gate is not an id oracle."""
        missing = self._verdict(seeded_pool, agent_policy, "get_order_status", 999999)
        foreign = self._verdict(seeded_pool, agent_policy, "get_order_status", 10003)
        assert missing.reason_code == foreign.reason_code

    def test_the_customers_own_order_passes_ownership(
        self, seeded_pool: SqliteConnectionPool, agent_policy: Any
    ) -> None:
        result = self._verdict(seeded_pool, agent_policy, "get_order_status", 10001)
        assert result.verdict == "allow"
        assert any(
            check.reason_code == "ORDER_OWNED_BY_CUSTOMER" for check in result.checks
        )

    def test_an_unauthenticated_read_is_refused(
        self, seeded_pool: SqliteConnectionPool, agent_policy: Any
    ) -> None:
        result = self._rules(seeded_pool, agent_policy).evaluate(
            ToolCall(action="get_order_status", params={"order_id": 10001}),
            AuthorityContext(actor="agent", granted_by="user_task", request_context={}),
        )
        assert result.verdict != "allow"
        assert result.reason_code == "CUSTOMER_NOT_AUTHENTICATED"

    def test_an_action_naming_no_order_is_still_not_applicable(
        self, seeded_pool: SqliteConnectionPool, agent_policy: Any
    ) -> None:
        """Ownership applies to orders; it must not start gating everything."""
        result = self._rules(seeded_pool, agent_policy).evaluate(
            ToolCall(action="list_orders", params={}),
            AuthorityContext(
                actor="agent", granted_by="user_task", request_context={"customer_id": 1}
            ),
        )
        assert result.verdict == "allow"


class TestTheUngovernedPathSkipsTheExecutionBoundary:
    """`gatekeeper: false` removes the governed path, not just its verdict.

    The Gatekeeper decides; the Executor bounds. With the feature
    withheld the graph calls the agent's domain adapter directly, so
    neither stage appears in the trace — the absence *is* the
    demonstration, and a GATEKEEPER line reading "no gatekeeper" would be
    the screen describing a control that is not there.

    What this removes is the execution boundary: the executor's
    idempotency key, its deadline, and its normalized outcome. It does
    not remove every guard beneath it — `RefundSaga` keeps its own
    idempotency, and the repositories keep scoping to the authenticated
    customer. See the last test here, which pins that down.
    """

    class _Feed:
        def __init__(self) -> None:
            self.events: list[Any] = []

        def publish(self, event: Any) -> None:
            self.events.append(event)

        def emit(self, event: Any) -> None:
            self.events.append(event)

    async def _stages(self, on: bool, pool: SqliteConnectionPool) -> tuple[list[str], list[str]]:
        from agentic_commons.observability import StageReporter

        feed = self._Feed()
        agent = build_agent(
            llm=ScriptedLLM(
                [
                    {"tool_calls": tool_call("support.get_order_status", order_id=10001)},
                    {"content": "Order 10001 was delivered."},
                ]
            ),
            pool=pool,
            features=get_settings().features.with_all(True).with_override("gatekeeper", on),
            reporter=StageReporter(feed, "test"),
        )
        state = await agent.handle("what happened with order 10001?", session_id="s")
        return [str(getattr(e, "stage", "")) for e in feed.events], state.tool_results

    async def test_neither_stage_narrates_when_the_gatekeeper_is_withheld(
        self, seeded_pool: SqliteConnectionPool, agent_policy: Any, restore_settings: None
    ) -> None:
        stages, _ = await self._stages(False, seeded_pool)
        assert "gatekeeper_evaluation" not in stages
        assert "executor" not in stages

    async def test_the_gatekeeper_still_narrates_when_it_is_declared(
        self, seeded_pool: SqliteConnectionPool, agent_policy: Any, restore_settings: None
    ) -> None:
        stages, _ = await self._stages(True, seeded_pool)
        assert "gatekeeper_evaluation" in stages

    async def test_the_tool_still_runs_ungoverned(
        self, seeded_pool: SqliteConnectionPool, agent_policy: Any, restore_settings: None
    ) -> None:
        """Ungoverned means unchecked, not inert."""
        _, results = await self._stages(False, seeded_pool)
        assert results and "10001" in results[0]

    def test_the_saga_keeps_its_own_idempotency(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        """What dropping the execution boundary does *not* remove.

        The executor's idempotency key is gone on this path, but
        `RefundSaga` holds a second one keyed on order plus amount. A
        repeated refund is still a replay, not a second payment — so this
        is not a double-spend demonstration, and should not be sold as one.
        """
        from agentic_commons.persistence import SqlitePlatformStore
        from agentic_commons.resilience import SqliteIdempotencyGuard
        from customer_support_agent.persistence.repositories import RefundLedgerRepository
        from customer_support_agent.resilience.refund_saga import RefundSaga

        store = SqlitePlatformStore(seeded_pool, agent_id=get_settings().agent.id)
        saga = RefundSaga(
            OrdersRepository(seeded_pool),
            RefundLedgerRepository(seeded_pool),
            idempotency=SqliteIdempotencyGuard(store),
        )
        orders = OrdersRepository(seeded_pool)
        order = orders.get(10001)
        assert order is not None
        first = asyncio.run(saga.issue(order, 32.0, "damaged"))
        second = asyncio.run(saga.issue(orders.get(10001), 32.0, "damaged"))  # type: ignore[arg-type]
        assert first.succeeded and second.succeeded
        assert orders.get(10001).refund_status == "completed"  # type: ignore[union-attr]


class TestWritesAreScopedToTheEstablishedOrder:
    """Found in a live run, twice, in opposite directions.

    The scope used to be decided by matching the customer's message
    against a list of refund phrases. `"redunf for order 10979"` — one
    typo — matched nothing and a correct refund was denied. The same
    matcher granted a full refund mandate for `"do not refund order
    10979"`, because the bare word "refund" was in the list.

    Reading a sentence is the model's job now. What the Gatekeeper
    enforces is the part that needs no language at all: a write may only
    act on the order this conversation established, bounded by that
    order's authoritative total.
    """

    @staticmethod
    def _scope(seeded_pool: SqliteConnectionPool) -> EstablishedOrderScope:
        return EstablishedOrderScope(OrdersRepository(seeded_pool))

    @staticmethod
    def _proposal(order_id: int, amount: float) -> ToolCall:
        return ToolCall(
            action="support.queue_for_approval",
            params={"order_id": order_id, "amount": amount},
            required_permission="process_refunds",
            side_effecting=True,
        )

    def _covers(
        self, seeded_pool: SqliteConnectionPool, query: str, established: int | None,
        order_id: int = 10001, amount: float = 32.00,
    ) -> bool:
        context = {} if established is None else {ORDER_UNDER_DISCUSSION: established}
        scope = MandateSet(mandates=self._scope(seeded_pool).mandates_for("1", query, 1, context))
        return (
            scope.covering(
                self._proposal(order_id, amount), f"order:{order_id}", Money.parse(amount), 1
            )
            is not None
        )

    @pytest.mark.parametrize(
        "query",
        [
            "redunf for order 10001",   # the typo that was denied live
            "refund for order 10001",
            "send it back please",      # a refund request with no keyword at all
            "do not refund order 10001",  # the model must decide this, not a substring
            "",
        ],
    )
    def test_phrasing_no_longer_decides_authorization(
        self, seeded_pool: SqliteConnectionPool, query: str
    ) -> None:
        """Whatever the wording, the established order is in scope."""
        assert self._covers(seeded_pool, query, established=10001) is True

    def test_a_different_owned_order_is_out_of_scope(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        """Ownership is not enough: it must be the order under discussion.

        10002 belongs to the same customer and would pass every business
        rule. Only provenance stops the wrong refund.
        """
        assert self._covers(
            seeded_pool, "refund it", established=10001, order_id=10002, amount=210.00
        ) is False

    def test_no_established_order_means_no_write(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        assert self._covers(seeded_pool, "refund it", established=None) is False

    def test_the_amount_is_bounded_by_the_authoritative_total(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        assert self._covers(seeded_pool, "refund it", established=10001, amount=500.00) is False

    def test_another_customers_order_is_out_of_scope(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        """Ownership is re-read under the caller's own id, never carried."""
        assert self._covers(
            seeded_pool, "refund it", established=10003, order_id=10003, amount=45.00
        ) is False

    def test_the_proposed_call_cannot_supply_its_own_target(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        """`tool_params` is in metadata by the time a call is proposed.

        Reading it here would let a proposal nominate what it acts on,
        which is the whole thing this check exists to prevent.
        """
        scope = MandateSet(mandates=self._scope(seeded_pool).mandates_for(
            "1", "refund it", 1, {"tool_params": {"order_id": 10001}}
        ))
        assert scope.mandates == []
