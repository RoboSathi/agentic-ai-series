"""DeterministicCheck: the platform's contract, the agent's rules (LLD §21).

The split is the point. `agent_platform` owns `DeterministicChecker` — a
request in, findings out — and knows nothing about orders or refunds.
Customer Support supplies `OrderChecks`. Another agent declares the
same feature and injects its own checker, changing nothing in the
platform. These tests assert both halves, because a contract that quietly
grew domain vocabulary would still pass every behavioural test this agent
has.
"""

from typing import Any

from agent_platform.config import get_settings
from agent_platform.core.persistence import SqliteConnectionPool
from agent_platform.deterministic_checks import (
    AdmittedChecker,
    DeterministicChecker,
    DeterministicCheckRequest,
    DeterministicCheckResult,
)
from customer_support_agent.deterministic_checks.order_validator import OrderChecks
from customer_support_agent.graph import build_agent
from customer_support_agent.persistence.repositories import OrdersRepository
from tests.conftest import ScriptedLLM


class _AnotherAgentsRules:
    """A second agent's business logic, speaking none of Customer Support's."""

    def check(self, request: DeterministicCheckRequest) -> DeterministicCheckResult:
        if request.subject_id == "":
            return DeterministicCheckResult(passed=False, reason_code="NO_SUBJECT")
        return DeterministicCheckResult(passed=True, reason_code="TENANT_IN_GOOD_STANDING")


class TestThePlatformOwnsOnlyTheContract:
    def test_any_object_with_check_satisfies_the_protocol(self) -> None:
        assert isinstance(_AnotherAgentsRules(), DeterministicChecker)
        assert isinstance(OrderChecks(orders=None), DeterministicChecker)  # type: ignore[arg-type]

    def test_the_request_carries_no_domain_vocabulary(self) -> None:
        """`customer_id` here would oblige every agent to have customers."""
        fields = set(DeterministicCheckRequest.model_fields)
        assert fields == {"query", "agent_id", "session_id", "actor", "subject_id", "metadata"}

    def test_a_second_agent_can_supply_its_own_rules(
        self, seeded_pool: SqliteConnectionPool, restore_settings: None
    ) -> None:
        """States the posture it needs rather than inheriting the repo's.

        Whoever is working here may have the feature switched off in
        their own config; that is a deployment choice, not something a
        test should depend on.
        """
        agent = build_agent(
            llm=ScriptedLLM(),
            pool=seeded_pool,
            deterministic_checks=_AnotherAgentsRules(),
            features=get_settings().features.with_override("deterministic_checks", True),
        )
        # Wrapped so its facts enter as tokens (ADR P-40); the rules are still the agent's own.
        assert isinstance(agent._deterministic_checks, AdmittedChecker)
        assert isinstance(agent._deterministic_checks.inner, _AnotherAgentsRules)


class TestCustomerSupportRules:
    def _check(
        self, pool: SqliteConnectionPool, query: str, subject: str
    ) -> DeterministicCheckResult:
        return OrderChecks(OrdersRepository(pool)).check(
            DeterministicCheckRequest(query=query, subject_id=subject)
        )

    def test_an_unidentified_subject_fails_closed(self, seeded_pool: SqliteConnectionPool) -> None:
        result = self._check(seeded_pool, "refund order 10001", subject="")
        assert not result.passed and result.reason_code == "CUSTOMER_NOT_AUTHENTICATED"

    def test_another_customers_order_is_not_found(self, seeded_pool: SqliteConnectionPool) -> None:
        result = self._check(seeded_pool, "refund order 10001", subject="2")
        assert not result.passed and result.reason_code == "ORDER_NOT_FOUND_FOR_CUSTOMER"

    def test_a_message_with_no_order_reference_defers_rather_than_guessing(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        result = self._check(seeded_pool, "I'd like some help please", subject="1")
        assert result.passed and result.reason_code == "ORDER_REFERENCE_NOT_PRESENT"
        assert [f.applicable for f in result.findings] == [False]


class TestTheFeature:
    """Withheld, it costs a model call — it does not open a hole.

    The same predicates are re-asked by the business rules, the tools and
    the review queue. What disappears is the cheap early stop, which is
    exactly what the episode is demonstrating.
    """

    def _agent(self, on: bool, pool: SqliteConnectionPool) -> Any:
        settings = get_settings()
        return build_agent(
            llm=ScriptedLLM([{"content": "Order 10001 was delivered."}]),
            pool=pool,
            features=settings.features.with_override("deterministic_checks", on),
        )

    def test_the_stage_is_built_when_declared(
        self, seeded_pool: SqliteConnectionPool, restore_settings: None
    ) -> None:
        agent = self._agent(True, seeded_pool)
        assert agent._deterministic_checks is not None
        assert "deterministic_checks" in agent._graph.get_graph().nodes

    def test_the_stage_is_absent_when_withheld(
        self, seeded_pool: SqliteConnectionPool, restore_settings: None
    ) -> None:
        agent = self._agent(False, seeded_pool)
        assert agent._deterministic_checks is None
        assert "deterministic_checks" not in agent._graph.get_graph().nodes

    async def test_the_turn_still_completes_without_it(
        self, seeded_pool: SqliteConnectionPool, restore_settings: None
    ) -> None:
        state = await self._agent(False, seeded_pool).handle("what is order 10001?")
        assert state.response
