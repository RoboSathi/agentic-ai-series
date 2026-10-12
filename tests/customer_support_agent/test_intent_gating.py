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

from agent_platform.core.persistence import SqliteConnectionPool
from agent_platform.tools import NamespacedToolRegistry, ToolDefinition
from customer_support_agent.graph import build_agent
from customer_support_agent.intent.router import CustomerSupportIntentRouter
from tests.conftest import ScriptedLLM

MONEY_MOVING = {"process_refund", "queue_for_approval"}


class _Args(BaseModel):
    value: str = ""


class TestIntentClassification:
    def test_an_order_enquiry_unlocks_no_money_moving_tool(self) -> None:
        """The reported bug, stated as a property."""
        intent = CustomerSupportIntentRouter().classify("what is the status of order 10005")
        assert set(intent.allowed_tools).isdisjoint(MONEY_MOVING)

    def test_a_general_enquiry_unlocks_nothing(self) -> None:
        assert CustomerSupportIntentRouter().classify("who are you").allowed_tools == []


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

    def test_the_schema_sent_to_the_provider_uses_the_model_name(self) -> None:
        """The model reads its bare name; the namespace stays internal (ADR 0010)."""
        tool = ToolDefinition(
            name="process_refund",
            namespace="support",
            description="x",
            args_schema=_Args,
            handler=lambda **_: None,
        )
        assert tool.to_openai_schema()["function"]["name"] == "process_refund"

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
        from customer_support_agent.presentation.reporting import is_customer_ready

        leaked = '{"name": "support.get_order_status", "parameters": {"order_id": "10005"}}'
        assert is_customer_ready(leaked) is False

    def test_ordinary_prose_is_an_answer(self) -> None:
        from customer_support_agent.presentation.reporting import is_customer_ready

        assert is_customer_ready("Order 10005 was delivered on March 3.") is True

    def test_an_empty_reply_is_not_an_answer(self) -> None:
        from customer_support_agent.presentation.reporting import is_customer_ready

        assert is_customer_ready("   ") is False

    def test_prose_that_merely_mentions_braces_is_still_an_answer(self) -> None:
        from customer_support_agent.presentation.reporting import is_customer_ready

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
