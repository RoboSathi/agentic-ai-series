"""Regression: an unrelated query must not trigger a side-effecting call.

Found in a live run. "who are you?" caused the agent to look up an
arbitrary order and queue it for a refund, and "what's the status of
order 10042?" caused it to queue that order for approval without anyone
asking for a refund.

Four distinct defects, each tested below:

  1. Tool selection fell back to the *entire* toolset when nothing
     cleared the relevance bar — so an empty selection offered more than
     a narrow one.
  2. The relevance floor was low enough that a conversational message
     surfaced a money-moving tool.
  3. A tool the model was never offered could still be invoked, because
     lookup went to the full registry.
  4. Nothing stopped the model repeating an identical call, so one
     request produced two attempts at the same side effect.
"""

import pytest

from agentic_commons.config import get_settings
from agentic_commons.evals import create_scoring_cascade
from agentic_commons.persistence import SqliteConnectionPool
from agentic_commons.tools import NamespacedToolRegistry, ToolCall, ToolRetrievalSelector
from customer_support_agent.control_loop import DuplicateActionDetector, action_fingerprint
from customer_support_agent.core import HitlQueueRepository, OrdersRepository, build_agent
from customer_support_agent.core.refund_saga import RefundSaga
from customer_support_agent.core.repositories import RefundLedgerRepository
from customer_support_agent.core.tools import SupportTools
from tests.conftest import ScriptedLLM, tool_call

# A message with no order and no refund intent.
CONVERSATIONAL = "who are you ?"


@pytest.fixture
def selector(seeded_pool: SqliteConnectionPool) -> ToolRetrievalSelector:
    orders = OrdersRepository(seeded_pool)
    refunds = RefundLedgerRepository(seeded_pool)
    tools = SupportTools(
        orders, HitlQueueRepository(seeded_pool), refunds, RefundSaga(orders, refunds)
    )
    registry = NamespacedToolRegistry()
    registry.register_all(tools.definitions())
    return ToolRetrievalSelector(registry, create_scoring_cascade())


class TestSelectionOffersNothingRatherThanEverything:
    async def test_a_conversational_query_surfaces_no_tools(
        self, selector: ToolRetrievalSelector
    ) -> None:
        """Defect 1 and 2: the empty selection used to mean 'offer them all'."""
        assert await selector.select(CONVERSATIONAL) == []

    async def test_no_money_moving_tool_is_offered_for_chat(
        self, selector: ToolRetrievalSelector
    ) -> None:
        offered = {tool.name for tool in await selector.select(CONVERSATIONAL)}
        assert offered.isdisjoint({"process_refund"})

    async def test_a_status_query_offers_only_read_only_tools(
        self, selector: ToolRetrievalSelector
    ) -> None:
        """Asking about an order is not asking to refund it."""
        offered = {tool.name for tool in await selector.select("what is the status of order 10001")}
        assert "process_refund" not in offered

    async def test_a_refund_request_still_offers_the_refund_tools(
        self, selector: ToolRetrievalSelector
    ) -> None:
        """The narrowing must not break the case the agent exists for."""
        offered = {tool.name for tool in await selector.select("I want a refund for order 10001")}
        assert "process_refund" in offered

    def test_the_relevance_floor_is_configured_not_hardcoded(self) -> None:
        assert get_settings().tools.selection_min_score > 0.1


class TestOfferedSetIsEnforcedAtInvocation:
    async def test_a_tool_that_was_not_offered_cannot_be_called(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        """Defect 3: narrowing must be control, not presentation.

        The model names a refund tool while asking a conversational
        question, so it was never offered one. The call must not reach
        the tool even though the registry knows it.
        """
        llm = ScriptedLLM(
            [
                {"tool_calls": tool_call("support.process_refund", order_id=10001, amount=32.0)},
                {"content": "I'm a support agent."},
            ]
        )
        agent = build_agent(llm=llm, pool=seeded_pool, with_memory=False)
        state = await agent.handle(CONVERSATIONAL)
        assert OrdersRepository(seeded_pool).get(10001).refund_status == "none"  # type: ignore[union-attr]
        assert any("not offered" in decision for decision in state.decisions)

    async def test_no_refund_is_queued_for_a_conversational_query(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        """The exact live failure: 'who are you?' queued a refund."""
        llm = ScriptedLLM(
            [
                {
                    "tool_calls": tool_call(
                        "support.queue_for_approval", order_id=10002, amount=210.0
                    )
                },
                {"content": "I'm a support agent."},
            ]
        )
        agent = build_agent(llm=llm, pool=seeded_pool, with_memory=False)
        await agent.handle(CONVERSATIONAL)
        assert HitlQueueRepository(seeded_pool).pending() == []


class TestDuplicateActionSuppression:
    def test_identical_calls_share_a_fingerprint(self) -> None:
        first = ToolCall(action="support.get_order_status", params={"order_id": "1"})
        second = ToolCall(action="support.get_order_status", params={"order_id": "1"})
        assert action_fingerprint(first) == action_fingerprint(second)

    def test_argument_order_does_not_change_identity(self) -> None:
        """Order varies between turns and means nothing."""
        first = ToolCall(action="x", params={"a": 1, "b": 2})
        second = ToolCall(action="x", params={"b": 2, "a": 1})
        assert action_fingerprint(first) == action_fingerprint(second)

    def test_different_arguments_are_different_calls(self) -> None:
        first = ToolCall(action="x", params={"order_id": "1"})
        second = ToolCall(action="x", params={"order_id": "2"})
        assert action_fingerprint(first) != action_fingerprint(second)

    def test_a_repeat_is_answered_with_the_earlier_result(self) -> None:
        """Refusing without the value invites a third attempt."""
        detector = DuplicateActionDetector()
        call = ToolCall(action="x", params={"order_id": "1"})
        detector.record(call, "order 1 is delivered")
        assert "order 1 is delivered" in detector.reject(call)

    async def test_the_agent_executes_a_repeated_call_only_once(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        """Defect 4: one request produced two attempts at the same effect."""
        llm = ScriptedLLM(
            [
                {
                    "tool_calls": tool_call(
                        "support.queue_for_approval", order_id=10002, amount=210.0
                    )
                },
                {
                    "tool_calls": tool_call(
                        "support.queue_for_approval", order_id=10002, amount=210.0
                    )
                },
                {"content": "Queued for review."},
            ]
        )
        agent = build_agent(llm=llm, pool=seeded_pool, with_memory=False)
        state = await agent.handle("I want a refund for order 10002")
        assert len(HitlQueueRepository(seeded_pool).pending()) == 1
        assert any("suppressed" in decision for decision in state.decisions)

    async def test_the_detector_resets_between_runs(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        """A second customer asking the same thing is not a repeat."""
        script = [
            {"tool_calls": tool_call("support.get_order_status", order_id=10001)},
            {"content": "Delivered."},
        ]
        agent = build_agent(llm=ScriptedLLM(script), pool=seeded_pool, with_memory=False)
        await agent.handle("status of 10001", session_id="a")
        state = await agent.handle("status of 10001", session_id="b")
        assert not any("suppressed" in decision for decision in state.decisions)


class TestTheCustomerNeverSeesAHarnessError:
    async def test_a_tool_error_is_not_returned_as_the_answer(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        """`finalize` used to hand back the last message, tool errors included."""
        llm = ScriptedLLM(
            [
                {"tool_calls": tool_call("support.process_refund")},
                {"content": "I need the order number to help with that."},
            ]
        )
        agent = build_agent(llm=llm, pool=seeded_pool, with_memory=False)
        state = await agent.handle("refund my order please")
        assert "unavailable" not in state.response
        assert "Tool failed" not in state.response
