"""The Gatekeeper's four questions (LLD §11).

The truth table below is the heart of the suite: every combination of
surface, authority rank and cumulative state, asserted rather than
assumed. The four E12 live attacks live in the integration tests, where
they run against the whole assembled harness.
"""

from typing import Any

import pytest

from agent_platform.audit import NullAuditLedger, SqliteAuditLedger
from agent_platform.config import get_settings
from agent_platform.core.persistence import SqliteConnectionPool
from agent_platform.core.primitives.money import Money
from agent_platform.gatekeeper import (
    AuthorityContext,
    AuthorityHierarchyResolver,
    GatekeeperImpl,
)
from agent_platform.gatekeeper.ledger import CumulativeLedger
from agent_platform.gatekeeper.mandate import MandateSet
from agent_platform.gatekeeper.types import LedgerEntry
from agent_platform.runtime.executor import ApprovedToolCall
from agent_platform.security import Authorizer
from agent_platform.tools import ToolCall
from customer_support_agent.domain.order_reference import ORDER_UNDER_DISCUSSION
from customer_support_agent.gatekeeper.mandates import EstablishedOrderScope
from customer_support_agent.graph import build_agent
from customer_support_agent.persistence.repositories import OrdersRepository
from tests.conftest import ScriptedLLM


@pytest.fixture
def gatekeeper(seeded_pool: SqliteConnectionPool, agent_policy: Any) -> GatekeeperImpl:
    return GatekeeperImpl(
        authorizer=Authorizer(policy=agent_policy),
        policy=agent_policy,
        ledger=CumulativeLedger(seeded_pool, enabled=True),
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

    def test_outranks_compares_two_sources(self) -> None:
        resolver = AuthorityHierarchyResolver()
        assert resolver.outranks("provider", "user_task") is True
        assert resolver.outranks("tool_observations", "user_task") is False


class TestCumulativeLedger:
    def test_accumulates_within_the_window(self, pool: SqliteConnectionPool) -> None:
        ledger = CumulativeLedger(pool, enabled=True)
        for _ in range(3):
            ledger.record(
                LedgerEntry(ledger_key="a", action_class="refund", actor="a", amount=100.0)
            )
        assert ledger.window_total("a", "refund") == Money.parse("300.00")

    def test_budgets_are_per_key(self, pool: SqliteConnectionPool) -> None:
        ledger = CumulativeLedger(pool, enabled=True)
        ledger.record(LedgerEntry(ledger_key="a", action_class="refund", actor="a", amount=4000.0))
        assert ledger.window_total("b", "refund") == Money.zero()

    def test_an_unconfigured_class_is_uncapped(self, pool: SqliteConnectionPool) -> None:
        assert CumulativeLedger(pool).threshold_for("invented") is None


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
        self,
        seeded_pool: SqliteConnectionPool,
        query: str,
        established: int | None,
        order_id: int = 10001,
        amount: float = 32.00,
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
            "redunf for order 10001",  # the typo that was denied live
            "refund for order 10001",
            "send it back please",  # a refund request with no keyword at all
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
        assert (
            self._covers(seeded_pool, "refund it", established=10001, order_id=10002, amount=210.00)
            is False
        )

    def test_no_established_order_means_no_write(self, seeded_pool: SqliteConnectionPool) -> None:
        assert self._covers(seeded_pool, "refund it", established=None) is False

    def test_the_amount_is_bounded_by_the_authoritative_total(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        assert self._covers(seeded_pool, "refund it", established=10001, amount=500.00) is False

    def test_another_customers_order_is_out_of_scope(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        """Ownership is re-read under the caller's own id, never carried."""
        assert (
            self._covers(seeded_pool, "refund it", established=10003, order_id=10003, amount=45.00)
            is False
        )

    def test_the_proposed_call_cannot_supply_its_own_target(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        """`tool_params` is in metadata by the time a call is proposed.

        Reading it here would let a proposal nominate what it acts on,
        which is the whole thing this check exists to prevent.
        """
        scope = MandateSet(
            mandates=self._scope(seeded_pool).mandates_for(
                "1", "refund it", 1, {"tool_params": {"order_id": 10001}}
            )
        )
        assert scope.mandates == []
