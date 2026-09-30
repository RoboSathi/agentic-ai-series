"""BasicSanity: the platform's contract, the agent's rules (LLD §21).

The split is the point. `agentic_commons` owns `BasicSanityChecker` — a
request in, findings out — and knows nothing about orders or refunds.
Customer Support supplies `OrderSanityCheck`. Another agent declares the
same feature and injects its own checker, changing nothing in the
platform. These tests assert both halves, because a contract that quietly
grew domain vocabulary would still pass every behavioural test this agent
has.
"""

from typing import Any

from agentic_commons.basic_sanity import (
    BasicSanityChecker,
    BasicSanityRequest,
    BasicSanityResult,
)
from agentic_commons.config import get_settings
from agentic_commons.persistence import SqliteConnectionPool
from customer_support_agent.basic_sanity.order_validator import OrderSanityCheck
from customer_support_agent.graph import build_agent
from customer_support_agent.persistence.repositories import OrdersRepository
from tests.conftest import ScriptedLLM


class _AnotherAgentsRules:
    """A second agent's business logic, speaking none of Customer Support's."""

    def check(self, request: BasicSanityRequest) -> BasicSanityResult:
        if request.subject_id == "":
            return BasicSanityResult(passed=False, reason_code="NO_SUBJECT")
        return BasicSanityResult(passed=True, reason_code="TENANT_IN_GOOD_STANDING")


class TestThePlatformOwnsOnlyTheContract:
    def test_any_object_with_check_satisfies_the_protocol(self) -> None:
        assert isinstance(_AnotherAgentsRules(), BasicSanityChecker)
        assert isinstance(OrderSanityCheck(orders=None), BasicSanityChecker)  # type: ignore[arg-type]

    def test_the_request_carries_no_domain_vocabulary(self) -> None:
        """`customer_id` here would oblige every agent to have customers."""
        fields = set(BasicSanityRequest.model_fields)
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
            basic_sanity=_AnotherAgentsRules(),
            features=get_settings().features.with_override("basic_sanity", True),
        )
        assert isinstance(agent._basic_sanity, _AnotherAgentsRules)


class TestCustomerSupportRules:
    def _check(self, pool: SqliteConnectionPool, query: str, subject: str) -> BasicSanityResult:
        return OrderSanityCheck(OrdersRepository(pool)).check(
            BasicSanityRequest(query=query, subject_id=subject)
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
            features=settings.features.with_override("basic_sanity", on),
        )

    def test_the_stage_is_built_when_declared(
        self, seeded_pool: SqliteConnectionPool, restore_settings: None
    ) -> None:
        agent = self._agent(True, seeded_pool)
        assert agent._basic_sanity is not None
        assert "basic_sanity" in agent._graph.get_graph().nodes

    def test_the_stage_is_absent_when_withheld(
        self, seeded_pool: SqliteConnectionPool, restore_settings: None
    ) -> None:
        agent = self._agent(False, seeded_pool)
        assert agent._basic_sanity is None
        assert "basic_sanity" not in agent._graph.get_graph().nodes

    async def test_the_turn_still_completes_without_it(
        self, seeded_pool: SqliteConnectionPool, restore_settings: None
    ) -> None:
        state = await self._agent(False, seeded_pool).handle("what is order 10001?")
        assert state.response
