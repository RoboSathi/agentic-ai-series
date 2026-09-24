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
from customer_support_agent.graph import build_agent
from customer_support_agent.intent.router import CustomerSupportIntentRouter
from customer_support_agent.persistence import HitlQueueRepository, OrdersRepository
from tests.conftest import ScriptedLLM, tool_call

MONEY_MOVING = {"process_refund", "queue_for_approval"}


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
        assert CustomerSupportIntentRouter().classify(message).name == expected

    def test_a_refund_request_wins_over_the_order_words_it_contains(self) -> None:
        """ "a refund for order 10005" is a refund request, not an enquiry."""
        intent = CustomerSupportIntentRouter().classify("I want a refund for order 10005")
        assert intent.name == "refund_request"

    def test_an_order_enquiry_unlocks_no_money_moving_tool(self) -> None:
        """The reported bug, stated as a property."""
        permitted = CustomerSupportIntentRouter().permitted_tools("what is the status of order 10005")
        assert permitted.isdisjoint(MONEY_MOVING)

    def test_an_order_enquiry_still_unlocks_the_lookups(self) -> None:
        permitted = CustomerSupportIntentRouter().permitted_tools("status of order 10005")
        assert "get_order_status" in permitted

    def test_a_general_enquiry_unlocks_nothing(self) -> None:
        assert CustomerSupportIntentRouter().permitted_tools("who are you") == set()

    def test_a_refund_request_unlocks_the_refund_path(self) -> None:
        permitted = CustomerSupportIntentRouter().permitted_tools("I want a refund for order 10005")
        assert MONEY_MOVING <= permitted

    def test_matching_is_case_insensitive(self) -> None:
        assert CustomerSupportIntentRouter().classify("I WANT A REFUND").name == "refund_request"

    def test_the_matched_phrase_is_reported(self) -> None:
        """An operator should see *why* the agent was given refund tools."""
        assert CustomerSupportIntentRouter().classify("I want a refund").matched_phrase == "refund"


class TestTheAgentHonoursTheGate:
    async def test_a_status_query_never_queues_a_refund(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        """The exact reported failure, end to end."""
        llm = ScriptedLLM(
            [
                {"tool_calls": tool_call("support.get_order_status", order_id=10002)},
                {
                    "tool_calls": tool_call(
                        "support.queue_for_approval", order_id=10002, amount=210.0
                    )
                },
                {"content": "Order 10002 has been delivered."},
            ]
        )
        agent = build_agent(llm=llm, pool=seeded_pool, with_memory=False)
        state = await agent.handle("what is the status of order 10002")
        assert HitlQueueRepository(seeded_pool).pending() == []
        assert OrdersRepository(seeded_pool).get(10002).refund_status == "none"  # type: ignore[union-attr]
        assert any("not offered" in decision for decision in state.decisions)

    async def test_a_status_query_still_answers_from_a_lookup(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        """Gating must not break the question the customer actually asked."""
        llm = ScriptedLLM(
            [
                {"tool_calls": tool_call("support.get_order_status", order_id=10001)},
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
                {"tool_calls": tool_call("support.process_refund", order_id=10001, amount=32.0)},
                {"content": "Refunded."},
            ]
        )
        agent = build_agent(llm=llm, pool=seeded_pool, with_memory=False)
        await agent.handle("order 10001 was damaged, I want a refund")
        assert OrdersRepository(seeded_pool).get(10001).refund_status == "completed"  # type: ignore[union-attr]

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
        from customer_support_agent.graph.agent import is_customer_ready

        leaked = '{"name": "support.get_order_status", "parameters": {"order_id": "10005"}}'
        assert is_customer_ready(leaked) is False

    def test_ordinary_prose_is_an_answer(self) -> None:
        from customer_support_agent.graph.agent import is_customer_ready

        assert is_customer_ready("Order 10005 was delivered on March 3.") is True

    def test_an_empty_reply_is_not_an_answer(self) -> None:
        from customer_support_agent.graph.agent import is_customer_ready

        assert is_customer_ready("   ") is False

    def test_prose_that_merely_mentions_braces_is_still_an_answer(self) -> None:
        from customer_support_agent.graph.agent import is_customer_ready

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


class TestSemanticFallback:
    """The encoder runs second, and only where config allows it."""

    class StubEncoder:
        """A stand-in for the hosted encoder, with a fixed geometry.

        Every configured phrase sits on one axis. A query is placed on
        the same axis when `similar`, and orthogonal to it otherwise —
        so the similarity score is 1.0 or 0.0 and the assertions are
        about the *rule*, not about a hosted model's mood on the day.

        The phrases are recognised by their absence of a question mark,
        which every query in these tests carries.
        """

        _PHRASE_AXIS = [1.0, 0.0]
        _ELSEWHERE = [0.0, 1.0]

        def __init__(self, similar: bool = True) -> None:
            self.calls = 0
            self._similar = similar

        @property
        def dimensions(self) -> int:
            return 2

        async def embed(self, texts: list[str]) -> list[list[float]]:
            self.calls += 1
            return [self._vector_for(text) for text in texts]

        def _vector_for(self, text: str) -> list[float]:
            is_query = text.endswith("?")
            if not is_query:
                return self._PHRASE_AXIS
            return self._PHRASE_AXIS if self._similar else self._ELSEWHERE

    @pytest.mark.asyncio
    async def test_a_phrase_match_never_reaches_the_encoder(self) -> None:
        """Deterministic policy wins outright; the encoder is not consulted."""
        encoder = self.StubEncoder()
        intent = await CustomerSupportIntentRouter(embeddings=encoder).classify_async("where is my order")
        assert intent.name == "order_enquiry"
        assert encoder.calls == 0

    @pytest.mark.asyncio
    async def test_a_paraphrase_is_recognised_by_similarity(self) -> None:
        encoder = self.StubEncoder(similar=True)
        intent = await CustomerSupportIntentRouter(embeddings=encoder).classify_async("when will it get here?")
        assert intent.name == "order_enquiry"
        assert "similarity" in intent.matched_phrase

    @pytest.mark.asyncio
    async def test_a_weak_match_stays_on_the_default_intent(self) -> None:
        encoder = self.StubEncoder(similar=False)
        intent = await CustomerSupportIntentRouter(embeddings=encoder).classify_async("mattress thread count?")
        assert intent.name == "general_enquiry"
        assert not intent.permits_tools

    @pytest.mark.asyncio
    async def test_refund_is_never_reached_semantically(self) -> None:
        """Money moves on a stated policy, not on a threshold.

        `refund_request` is configured `semantic: false`, so however
        close a sentence sits to the refund vocabulary in embedding
        space, it cannot unlock the refund tools without saying so.
        """
        encoder = self.StubEncoder(similar=True)
        intent = await CustomerSupportIntentRouter(embeddings=encoder).classify_async("sort out the charge?")
        assert intent.name != "refund_request"
        assert "process_refund" not in intent.allowed_tools

    @pytest.mark.asyncio
    async def test_the_phrase_vocabulary_is_embedded_once(self) -> None:
        """The phrases cannot have moved, so re-embedding them is waste."""
        encoder = self.StubEncoder()
        classifier = CustomerSupportIntentRouter(embeddings=encoder)
        await classifier.classify_async("first question?")
        after_first = encoder.calls
        await classifier.classify_async("second question?")
        assert encoder.calls == after_first + 1

    @pytest.mark.asyncio
    async def test_an_unreachable_encoder_degrades_to_the_deterministic_answer(self) -> None:
        """A classifier that cannot reach its model must not fail the turn."""

        class Broken:
            @property
            def dimensions(self) -> int:
                return 2

            async def embed(self, texts: list[str]) -> list[list[float]]:
                raise RuntimeError("no entitlement")

        intent = await CustomerSupportIntentRouter(embeddings=Broken()).classify_async("something vague?")
        assert intent.name == "general_enquiry"
