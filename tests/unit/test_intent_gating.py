"""Refunds are discussed when the customer raises them (LLD §21).

The reported failure: asking for an order's *status* caused the agent to
queue that order for a refund. Embedding similarity was never the right
instrument for this — "is this about a refund?" became a threshold to
tune rather than a policy to state.

So intent is classified deterministically, before any score, and it
decides which tools exist for the turn. Retrieval still narrows within
that; it can never widen it.
"""

import pytest
from pydantic import BaseModel

from agentic_commons.persistence import SqliteConnectionPool
from agentic_commons.tools import NamespacedToolRegistry, ToolDefinition
from customer_support_agent.core import HitlQueueRepository, OrdersRepository, build_agent
from customer_support_agent.core.intent import IntentClassifier
from tests.conftest import ScriptedLLM, tool_call

MONEY_MOVING = {"process_refund", "apply_goodwill_credit", "queue_for_approval"}


class _Args(BaseModel):
    value: str = ""


class TestIntentClassification:
    @pytest.mark.parametrize(
        "message,expected",
        [
            ("what is the status of order 10005", "order_enquiry"),
            ("where is my parcel", "order_enquiry"),
            ("has order 10005 been delivered yet", "order_enquiry"),
            ("I want a refund for order 10005", "refund_request"),
            ("can I get my money back", "refund_request"),
            ("please compensate me for this", "refund_request"),
            ("who are you", "general_enquiry"),
            ("hello there", "general_enquiry"),
            ("what are your opening hours", "general_enquiry"),
        ],
    )
    def test_classifies_the_message(self, message: str, expected: str) -> None:
        assert IntentClassifier().classify(message).name == expected

    def test_a_refund_request_wins_over_the_order_words_it_contains(self) -> None:
        """ "a refund for order 10005" is a refund request, not an enquiry."""
        intent = IntentClassifier().classify("I want a refund for order 10005")
        assert intent.name == "refund_request"

    def test_an_order_enquiry_unlocks_no_money_moving_tool(self) -> None:
        """The reported bug, stated as a property."""
        permitted = IntentClassifier().permitted_tools("what is the status of order 10005")
        assert permitted.isdisjoint(MONEY_MOVING)

    def test_an_order_enquiry_still_unlocks_the_lookups(self) -> None:
        permitted = IntentClassifier().permitted_tools("status of order 10005")
        assert "get_order_status" in permitted

    def test_a_general_enquiry_unlocks_nothing(self) -> None:
        assert IntentClassifier().permitted_tools("who are you") == set()

    def test_a_refund_request_unlocks_the_refund_path(self) -> None:
        permitted = IntentClassifier().permitted_tools("I want a refund for order 10005")
        assert MONEY_MOVING <= permitted

    def test_matching_is_case_insensitive(self) -> None:
        assert IntentClassifier().classify("I WANT A REFUND").name == "refund_request"

    def test_the_matched_phrase_is_reported(self) -> None:
        """An operator should see *why* the agent was given refund tools."""
        assert IntentClassifier().classify("I want a refund").matched_phrase == "refund"


class TestTheAgentHonoursTheGate:
    async def test_a_status_query_never_queues_a_refund(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        """The exact reported failure, end to end."""
        llm = ScriptedLLM(
            [
                {"tool_calls": tool_call("support.get_order_status", order_id="10002")},
                {
                    "tool_calls": tool_call(
                        "support.queue_for_approval", order_id="10002", amount=210.0
                    )
                },
                {"content": "Order 10002 has been delivered."},
            ]
        )
        agent = build_agent(llm=llm, pool=seeded_pool, with_memory=False)
        state = await agent.handle("what is the status of order 10002")
        assert HitlQueueRepository(seeded_pool).pending() == []
        assert OrdersRepository(seeded_pool).get("10002").refund_status == "none"  # type: ignore[union-attr]
        assert any("not offered" in decision for decision in state.decisions)

    async def test_a_status_query_still_answers_from_a_lookup(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        """Gating must not break the question the customer actually asked."""
        llm = ScriptedLLM(
            [
                {"tool_calls": tool_call("support.get_order_status", order_id="10001")},
                {"content": "Order 10001 was delivered."},
            ]
        )
        agent = build_agent(llm=llm, pool=seeded_pool, with_memory=False)
        state = await agent.handle("what is the status of order 10001")
        assert any("get_order_status: ok" in decision for decision in state.decisions)
        assert state.response

    async def test_a_refund_request_still_reaches_the_refund_path(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        llm = ScriptedLLM(
            [
                {"tool_calls": tool_call("support.process_refund", order_id="10001", amount=32.0)},
                {"content": "Refunded."},
            ]
        )
        agent = build_agent(llm=llm, pool=seeded_pool, with_memory=False)
        await agent.handle("order 10001 was damaged, I want a refund")
        assert OrdersRepository(seeded_pool).get("10001").refund_status == "completed"  # type: ignore[union-attr]

    async def test_the_intent_is_recorded_for_the_operator(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        agent = build_agent(
            llm=ScriptedLLM([{"content": "hi"}]), pool=seeded_pool, with_memory=False
        )
        state = await agent.handle("who are you")
        assert any("intent: general_enquiry" in decision for decision in state.decisions)


class TestProviderLegalToolNames:
    def test_the_wire_name_carries_no_dot(self) -> None:
        """A provider rejects the whole request over one illegal name."""
        tool = ToolDefinition(
            name="get_order_status",
            namespace="support",
            description="x",
            args_schema=_Args,
            handler=lambda **_: None,
        )
        assert tool.wire_name == "support__get_order_status"
        assert "." not in tool.wire_name

    def test_the_schema_sent_to_the_provider_uses_the_wire_name(self) -> None:
        tool = ToolDefinition(
            name="process_refund",
            namespace="support",
            description="x",
            args_schema=_Args,
            handler=lambda **_: None,
        )
        assert tool.to_openai_schema()["function"]["name"] == "support__process_refund"

    def test_a_wire_name_resolves_back_to_its_tool(self) -> None:
        """The model answers with the name it was shown."""
        registry = NamespacedToolRegistry()
        registry.register(
            ToolDefinition(
                name="process_refund",
                namespace="support",
                description="x",
                args_schema=_Args,
                handler=lambda **_: None,
            )
        )
        assert registry.get("support__process_refund") is not None

    def test_an_illegal_tool_name_fails_at_registration(self) -> None:
        """Not halfway through a customer's first conversation."""
        registry = NamespacedToolRegistry()
        with pytest.raises(ValueError, match="valid provider function name"):
            registry.register(
                ToolDefinition(
                    name="get order status",
                    namespace="support",
                    description="x",
                    args_schema=_Args,
                    handler=lambda **_: None,
                )
            )


class TestMachineOutputNeverReachesTheCustomer:
    def test_a_tool_call_emitted_as_text_is_not_an_answer(self) -> None:
        """Seen live: the final reply was `{"name": ..., "parameters": ...}`."""
        from customer_support_agent.core.graph import is_customer_ready

        leaked = '{"name": "support.get_order_status", "parameters": {"order_id": "10005"}}'
        assert is_customer_ready(leaked) is False

    def test_ordinary_prose_is_an_answer(self) -> None:
        from customer_support_agent.core.graph import is_customer_ready

        assert is_customer_ready("Order 10005 was delivered on March 3.") is True

    def test_an_empty_reply_is_not_an_answer(self) -> None:
        from customer_support_agent.core.graph import is_customer_ready

        assert is_customer_ready("   ") is False

    def test_prose_that_merely_mentions_braces_is_still_an_answer(self) -> None:
        from customer_support_agent.core.graph import is_customer_ready

        assert is_customer_ready("Your order {reference} is on its way.") is True

    async def test_the_customer_gets_prose_not_json(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        llm = ScriptedLLM(
            [{"content": '{"name": "support.get_order_status", "parameters": {"order_id": "1"}}'}]
        )
        agent = build_agent(llm=llm, pool=seeded_pool, with_memory=False)
        state = await agent.handle("who are you")
        assert '"parameters"' not in state.response
