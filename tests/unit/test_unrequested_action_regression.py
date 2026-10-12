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

from agent_platform.config import get_settings
from agent_platform.context.similarity import LexicalSimilarity
from agent_platform.loop_control import LoopBudget, LoopGovernor, action_fingerprint
from agent_platform.persistence import SqliteConnectionPool
from agent_platform.scoring import create_scoring_cascade
from agent_platform.tools import NamespacedToolRegistry, ToolCall, ToolRetrievalSelector
from customer_support_agent.graph import build_agent
from customer_support_agent.persistence import HitlQueueRepository, OrdersRepository
from customer_support_agent.persistence.repositories import RefundLedgerRepository
from customer_support_agent.resilience.refund_saga import RefundSaga
from customer_support_agent.tools.definitions import SupportTools
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

    async def test_a_refund_request_still_offers_the_refund_tools(
        self, selector: ToolRetrievalSelector
    ) -> None:
        """The narrowing must not break the case the agent exists for."""
        offered = {tool.name for tool in await selector.select("I want a refund for order 10001")}
        assert "process_refund" in offered

    def test_the_relevance_floor_is_configured_not_hardcoded(self) -> None:
        assert get_settings().tools.selection_min_score > 0.1


class TestOfferedSetIsEnforcedAtInvocation:
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
        # A reply the check cut may be queued for review; no money request may be.
        assert [r for r in HitlQueueRepository(seeded_pool).pending() if not r.is_reply] == []


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

    async def test_a_repeat_is_answered_with_the_earlier_result(self) -> None:
        """Refusing without the value invites a third attempt."""
        governor = _governor()
        call = ToolCall(action="x", params={"order_id": "1"})
        await governor.before_action(call)
        await governor.after_action(call, "order 1 is delivered", True)
        verdict = await governor.before_action(call)
        assert verdict.decision == "suppress"
        assert "order 1 is delivered" in governor.suppression_message(verdict)

    async def test_the_same_call_written_differently_is_a_repeat(self) -> None:
        """Key order, case, whitespace and `5` vs "5" mean nothing."""
        governor = _governor()
        first = ToolCall(action="x", params={"Order_Id": "10001 ", "note": "Hi"})
        await governor.before_action(first)
        await governor.after_action(first, "delivered", True)
        again = ToolCall(action="x", params={"note": "hi", "order_id": 10001})
        verdict = await governor.before_action(again)
        assert verdict.decision == "suppress" and verdict.reason_code == "CANONICAL_DUPLICATE"

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


class TestARefusedCallIsNotRetried:
    """Found in a live run: one refusal became three, then the iteration cap.

    The Gatekeeper refused `queue_for_approval`; the model proposed it
    again, then a third time byte-identically, and only the round ceiling
    stopped it — three model round-trips and ~21 seconds spent on a
    verdict that could not change. Two causes, both fixed here: the
    duplicate detector only ever learned about calls that *succeeded*,
    and the refusal handed back to the model read like a transient error.
    """

    async def test_the_second_attempt_is_suppressed(self) -> None:
        governor = _governor()
        call = ToolCall(
            action="support.queue_for_approval",
            params={"order_id": 10979, "amount": 56.86},
            side_effecting=True,
        )
        assert (await governor.before_action(call)).decision == "continue"
        governor.record_refusal(call, "That action was not permitted: not in scope.")
        verdict = await governor.before_action(call)
        assert verdict.decision == "suppress"
        assert verdict.reason_code == "DUPLICATE_ACTION"

    async def test_the_model_is_told_the_refusal_is_final(self) -> None:
        """Not just that it failed — that trying again cannot help."""
        governor = _governor()
        call = ToolCall(
            action="support.process_refund", params={"order_id": 1}, side_effecting=True
        )
        await governor.before_action(call)
        governor.record_refusal(call, "not in scope")
        message = governor.suppression_message(await governor.before_action(call))
        assert "refused" in message.lower()
        assert "cannot change" in message.lower()

    async def test_a_different_call_is_still_allowed_after_a_refusal(self) -> None:
        """A refusal ends that proposal, not every remaining option."""
        governor = _governor()
        refused = ToolCall(
            action="support.process_refund", params={"order_id": 1}, side_effecting=True
        )
        await governor.before_action(refused)
        governor.record_refusal(refused, "not in scope")
        other = ToolCall(action="support.get_order_status", params={"order_id": 1})
        assert (await governor.before_action(other)).decision == "continue"


def _governor() -> LoopGovernor:
    governor = LoopGovernor(LoopBudget(max_steps=5), similarity=LexicalSimilarity())
    governor.begin_turn()
    governor.begin_step()
    return governor
