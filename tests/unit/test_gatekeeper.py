"""The Gatekeeper's four questions (LLD §11).

The truth table below is the heart of the suite: every combination of
surface, authority rank and cumulative state, asserted rather than
assumed. The four E12 live attacks live in the integration tests, where
they run against the whole assembled harness.
"""

import pytest

from agentic_commons.audit import SqliteAuditLedger
from agentic_commons.config import get_settings
from agentic_commons.gatekeeper import (
    AuthorityContext,
    AuthorityHierarchyResolver,
    AutomationLevelPolicy,
    GatekeeperImpl,
    SurfaceClassifier,
    ThreeGatesEvaluator,
)
from agentic_commons.gatekeeper.ledger import CumulativeLedger
from agentic_commons.gatekeeper.tier2 import NullTier2Advisor
from agentic_commons.gatekeeper.types import LedgerEntry
from agentic_commons.persistence import SqliteConnectionPool
from agentic_commons.security import Authorizer
from agentic_commons.tools import ToolCall


@pytest.fixture
def gatekeeper(pool: SqliteConnectionPool) -> GatekeeperImpl:
    return GatekeeperImpl(
        authorizer=Authorizer(),
        ledger=CumulativeLedger(pool),
        audit=SqliteAuditLedger(pool),
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


class TestSurfaceClassifier:
    @pytest.mark.parametrize(
        "action,surface",
        [
            ("get_order_status", "editable"),
            ("queue_for_approval", "append_only"),
            ("update_refund_policy", "locked"),
            ("approve_hitl_request", "human_controlled"),
        ],
    )
    def test_classifies_configured_actions(self, action: str, surface: str) -> None:
        assert SurfaceClassifier().classify(action) == surface

    def test_an_unclassified_action_defaults_to_human_controlled(self) -> None:
        assert SurfaceClassifier().classify("invented_action") == "human_controlled"

    def test_locked_surfaces_are_never_agent_writable(self) -> None:
        writable, reason = SurfaceClassifier().is_writable_by_agent("locked")
        assert writable is False
        assert "locked" in reason

    def test_namespaced_actions_resolve_to_their_bare_name(self) -> None:
        assert SurfaceClassifier().classify("support.process_refund") == "editable"


class TestThreeGates:
    def test_a_narrow_certifiable_action_passes(self) -> None:
        assert ThreeGatesEvaluator().evaluate("single", "single_record", True).passes

    def test_open_path_with_global_radius_fails(self) -> None:
        result = ThreeGatesEvaluator().evaluate("open", "global", True)
        assert result.passes is False
        assert "open-ended" in result.reason

    def test_uncertifiable_wide_action_fails(self) -> None:
        assert ThreeGatesEvaluator().evaluate("bounded", "tenant", False).passes is False


class TestAutomationLevels:
    def test_reads_the_configured_sheridan_level(self) -> None:
        assert AutomationLevelPolicy().level_for("get_order_status") == 10

    def test_an_amount_over_the_ceiling_requires_a_human(self) -> None:
        assessment = AutomationLevelPolicy().assess("process_refund", 200.0)
        assert assessment.requires_human is True

    def test_an_amount_under_the_ceiling_does_not(self) -> None:
        assert AutomationLevelPolicy().assess("process_refund", 20.0).requires_human is False

    def test_the_boundary_amount_is_permitted(self) -> None:
        """Exactly at the ceiling is within policy; only above it escalates."""
        ceiling = get_settings().gatekeeper.automation.per_action_amount_escalation_usd[
            "process_refund"
        ]
        assert AutomationLevelPolicy().assess("process_refund", ceiling).requires_human is False

    def test_an_unconfigured_action_needs_approval(self) -> None:
        assert AutomationLevelPolicy().assess("invented_action").requires_human is False


class TestCumulativeLedger:
    def test_maps_an_action_to_its_budget_class(self, pool: SqliteConnectionPool) -> None:
        assert CumulativeLedger(pool).class_of("support.process_refund") == "refund"

    def test_accumulates_within_the_window(self, pool: SqliteConnectionPool) -> None:
        ledger = CumulativeLedger(pool)
        for _ in range(3):
            ledger.record(
                LedgerEntry(ledger_key="a", action_class="refund", actor="a", amount=100.0)
            )
        assert ledger.window_total("a", "refund") == 300.0

    def test_detects_a_breach_before_it_happens(self, pool: SqliteConnectionPool) -> None:
        """The check asks 'would this exceed?', not 'did it?'."""
        ledger = CumulativeLedger(pool)
        ledger.record(LedgerEntry(ledger_key="a", action_class="refund", actor="a", amount=3990.0))
        assert ledger.would_exceed("a", "refund", 5.0) is False
        assert ledger.would_exceed("a", "refund", 50.0) is True

    def test_budgets_are_per_key(self, pool: SqliteConnectionPool) -> None:
        ledger = CumulativeLedger(pool)
        ledger.record(LedgerEntry(ledger_key="a", action_class="refund", actor="a", amount=4000.0))
        assert ledger.would_exceed("b", "refund", 100.0) is False

    def test_an_unconfigured_class_is_uncapped(self, pool: SqliteConnectionPool) -> None:
        assert CumulativeLedger(pool).threshold_for("invented") == float("inf")


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
        assert ledger.window_total("agent-1", "refund") == 0.0
        gatekeeper.commit(call, agent_authority)
        assert ledger.window_total("agent-1", "refund") == 25.0
