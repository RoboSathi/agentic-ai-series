"""How a support turn is routed before the model is asked (graph order, placement, reuse).

Built by the real composition root against the seeded database; the intent classifier
and the model are scripted, so these assert which stages ran, in what order, and whether
the model was asked at all.
"""

import pytest

from agent_platform.config import FeatureSet, get_settings
from agent_platform.intent import IntentCandidate
from agent_platform.observability import InMemoryLiveStatusFeed, StageReporter
from agent_platform.observability.types import StatusEvent
from agent_platform.persistence import SqliteConnectionPool
from customer_support_agent.evals.fixture import policy_seed_sql
from customer_support_agent.graph import build_agent
from customer_support_agent.intent.router import CustomerSupportIntentRouter
from customer_support_agent.persistence.repositories import HitlQueueRepository, OrdersRepository
from tests.conftest import ScriptedLLM, tool_call

# Names an order the seeded customer owns, in words no phrase rule places.
_UNPLACED_WITH_ORDER = "hmm, about 10001"
_UNPLACED = "hmm, something is off"
_HOSTED = ("memory.semantic", "memory.episodic", "hallucination_guard")
_SCREENS = ("prompt_injection", "content_moderation", "pii")


class _UnsureBackend:
    """A classifier that places nothing: every turn comes back UNKNOWN."""

    model_name = "scripted-unsure"
    score_type = "model_softmax"
    is_loaded = True

    async def ensure_loaded(self) -> None:
        return None

    async def classify(self, text: str) -> list[IntentCandidate]:
        return [IntentCandidate(intent="UNKNOWN", score=0.9)]


def _features(*withheld: str) -> FeatureSet:
    features = get_settings().features
    for path in (*_HOSTED, *withheld):
        features = features.with_override(path, False)
    return features


@pytest.fixture(autouse=True)
def deployed_policy(seeded_pool: SqliteConnectionPool) -> None:
    seeded_pool.execute_script(policy_seed_sql())


async def _turn(
    pool: SqliteConnectionPool, query: str, llm: ScriptedLLM, features: FeatureSet
) -> tuple[dict[str, object], list[StatusEvent]]:
    feed = InMemoryLiveStatusFeed(max_events=0)
    events: list[StatusEvent] = []
    feed.subscribe(events.append)
    agent = build_agent(
        llm=llm,
        pool=pool,
        features=features,
        intent_router=CustomerSupportIntentRouter(backend=_UnsureBackend()),
        reporter=StageReporter(feed, run_id="csa-call-flow"),
    )
    state = await agent.handle(query)
    return state.metadata, events


class TestStageOrder:
    def test_security_screens_run_before_the_deterministic_checks(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        """Untrusted text is vetted before anything reads the database on its behalf."""
        agent = build_agent(llm=ScriptedLLM([]), pool=seeded_pool, features=_features())
        edges = {(e.source, e.target) for e in agent._graph.get_graph().edges}
        assert ("content_moderation", "deterministic_checks") in edges
        assert ("deterministic_checks", "classify_intent") in edges
        assert ("choose_order", "pii_filtering") in edges


class TestAnUnplacedTurnThatNamesAnOrder:
    async def test_it_is_answered_as_that_orders_status_without_a_model_call(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        llm = ScriptedLLM([])
        features = _features("prompt_injection", "content_moderation", "pii")
        metadata, _ = await _turn(seeded_pool, _UNPLACED_WITH_ORDER, llm, features)
        assert metadata["intent"] == "ORDER_STATUS"
        assert metadata["intent_source"] == "order_reference"
        assert metadata["intent_execution_route"] == "deterministic_read"
        assert llm.requests == []

    async def test_one_that_names_no_order_stays_unknown(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        llm = ScriptedLLM([{"content": "Which order do you mean?"}])
        features = _features("prompt_injection", "content_moderation", "pii")
        metadata, _ = await _turn(seeded_pool, _UNPLACED, llm, features)
        assert metadata["intent"] == "UNKNOWN"
        assert metadata["intent_execution_route"] == "llm_fallback"


class TestAnOverLimitRefundGoesThroughTheGatekeeper:
    """Above the limit the model still calls `process_refund`; the Gatekeeper decides.

    Reported: the prompt told the model "do not call process_refund" above the limit,
    so a plain over-limit refund was declined by the model and never reached a person.
    """

    _OVER = 10002  # $210.00, above the seeded $50.00 limit

    async def _refund(self, pool: SqliteConnectionPool, features: FeatureSet) -> ScriptedLLM:
        llm = ScriptedLLM(
            [
                {"tool_calls": tool_call("process_refund", order_id=self._OVER, amount=210.0)},
                {"content": "Your refund request has been sent for review."},
            ]
        )
        agent = build_agent(llm=llm, pool=pool, features=features)
        await agent.handle(f"refund order {self._OVER}")
        return llm

    async def test_the_prompt_states_the_limit_but_never_forbids_the_call(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        llm = await self._refund(seeded_pool, _features(*_SCREENS))
        system = str(llm.requests[0].messages[0].content)
        assert "$50.00" in system and "still goes through process_refund" in system
        assert "do not call" not in system.casefold()
        assert [t["function"]["name"] for t in llm.requests[0].tools or []] == ["process_refund"]

    async def test_with_the_gatekeeper_it_is_queued_for_a_person(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        await self._refund(seeded_pool, _features(*_SCREENS))
        [queued] = HitlQueueRepository(seeded_pool).pending_for_user(1)
        assert (queued.order_id, queued.reason) == (self._OVER, "refund")
        order = OrdersRepository(seeded_pool).get(self._OVER)
        assert order is not None and order.refund_status == "pending_review"

    async def test_without_the_gatekeeper_the_money_moves(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        await self._refund(seeded_pool, _features(*_SCREENS, "gatekeeper"))
        assert HitlQueueRepository(seeded_pool).pending_for_user(1) == []
        order = OrdersRepository(seeded_pool).get(self._OVER)
        assert order is not None and order.refund_status == "issued"


class _SureItIsARefund(_UnsureBackend):
    """A classifier that reads every message as a refund request."""

    async def classify(self, text: str) -> list[IntentCandidate]:
        return [IntentCandidate(intent="REQUEST_REFUND", score=0.99)]


class TestARefusedRefundNamingAnOrder:
    async def test_it_is_answered_as_that_orders_status(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        """ "I don't want a refund for 10001" is about the order, not a refund request."""
        llm = ScriptedLLM([])
        agent = build_agent(
            llm=llm,
            pool=seeded_pool,
            features=_features(*_SCREENS),
            intent_router=CustomerSupportIntentRouter(backend=_SureItIsARefund()),
        )
        state = await agent.handle("I don't want a refund for order 10001")
        assert state.metadata["intent"] == "ORDER_STATUS"
        assert state.metadata["intent_execution_route"] == "deterministic_read"
        assert llm.requests == []
