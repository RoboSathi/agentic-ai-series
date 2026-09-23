"""Follow-up turns keep their subject (LLD §21).

Reported from a live chat:

    > status of order 10005      → answered correctly
    > what about 10010           → {"order_id": "10010", ...}
    > how about 10008            → {"tool": "get_order_status", ...}

Two defects. The agent held no conversation history, so a follow-up that
named no subject matched no intent and was offered no tools; and the
machine output the model produced instead was handed to the customer,
because the leak check only recognised the one JSON shape seen first.
"""

from agentic_commons.config import get_settings
from agentic_commons.persistence import SqliteConnectionPool
from customer_support_agent.config import get_agent_settings
from customer_support_agent.core import build_agent
from customer_support_agent.core.graph import is_customer_ready
from customer_support_agent.core.intent import CustomerSupportIntentRouter
from tests.conftest import ScriptedLLM, tool_call


class TestFollowupInheritsTheSubject:
    def test_a_bare_reference_continues_the_previous_intent(self) -> None:
        classifier = CustomerSupportIntentRouter()
        first = classifier.classify("status of order 10005")
        assert classifier.classify("what about 10010", previous=first).name == "order_enquiry"

    def test_a_bare_order_number_alone_continues_it(self) -> None:
        classifier = CustomerSupportIntentRouter()
        first = classifier.classify("status of order 10005")
        assert classifier.classify("10010?", previous=first).name == "order_enquiry"

    def test_the_continuation_is_marked_as_such(self) -> None:
        """An operator should see the intent was inherited, not matched."""
        classifier = CustomerSupportIntentRouter()
        first = classifier.classify("status of order 10005")
        assert classifier.classify("how about 10008", previous=first).matched_phrase == "continued"

    def test_an_unrelated_follow_up_does_not_inherit(self) -> None:
        """ "thanks, bye" is not a continuation of an order enquiry."""
        classifier = CustomerSupportIntentRouter()
        first = classifier.classify("status of order 10005")
        assert classifier.classify("thanks, goodbye", previous=first).name == "general_enquiry"

    def test_nothing_is_inherited_from_a_toolless_intent(self) -> None:
        classifier = CustomerSupportIntentRouter()
        first = classifier.classify("who are you")
        assert classifier.classify("what about 10010", previous=first).name == "general_enquiry"

    def test_inheritance_never_widens_authority(self) -> None:
        """A follow-up reuses the earlier intent; it cannot escalate it."""
        classifier = CustomerSupportIntentRouter()
        enquiry = classifier.classify("status of order 10005")
        continued = classifier.classify("what about 10010", previous=enquiry)
        assert "process_refund" not in continued.allowed_tools

    def test_an_explicit_refund_still_wins_over_inheritance(self) -> None:
        classifier = CustomerSupportIntentRouter()
        enquiry = classifier.classify("status of order 10005")
        assert classifier.classify("refund it then", previous=enquiry).name == "refund_request"


class TestConversationHistory:
    async def test_prior_turns_are_replayed_to_the_model(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        """Without them the follow-up's subject does not exist anywhere."""
        llm = ScriptedLLM([{"content": "Order 10001 was delivered."}])
        agent = build_agent(llm=llm, pool=seeded_pool, with_memory=False)
        await agent.handle("status of order 10001", session_id="s1")
        await agent.handle("what about 10002", session_id="s1")
        replayed = [message.content for message in llm.requests[-1].messages]
        assert "status of order 10001" in replayed

    async def test_history_is_scoped_to_its_session(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        """One customer's conversation must not leak into another's."""
        agent = build_agent(
            llm=ScriptedLLM([{"content": "ok"}]), pool=seeded_pool, with_memory=False
        )
        await agent.handle("status of order 10001", session_id="alice")
        assert agent.history_for("bob") == []

    async def test_history_is_bounded(self, seeded_pool: SqliteConnectionPool) -> None:
        """A conversation cannot grow the prompt without limit."""
        agent = build_agent(
            llm=ScriptedLLM([{"content": "ok"}]), pool=seeded_pool, with_memory=False
        )
        for index in range(20):
            await agent.handle(f"message {index}", session_id="s1")
        cap = get_agent_settings().history_turns * 2
        assert len(agent.history_for("s1")) <= cap

    async def test_a_follow_up_reaches_the_lookup_tool(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        """The reported failure, end to end."""
        llm = ScriptedLLM(
            [
                {"tool_calls": tool_call("support.get_order_status", order_id=10001)},
                {"content": "Order 10001 was delivered."},
                {"tool_calls": tool_call("support.get_order_status", order_id=10002)},
                {"content": "Order 10002 was delivered."},
            ]
        )
        agent = build_agent(llm=llm, pool=seeded_pool, with_memory=False)
        await agent.handle("status of order 10001", session_id="s1")
        state = await agent.handle("what about 10002", session_id="s1")
        assert any("get_order_status: ok" in decision for decision in state.decisions)


class TestMachineOutputIsNeverTheAnswer:
    def test_catches_every_shape_seen_live(self) -> None:
        """Three different key sets appeared; matching key names misses some."""
        for leaked in (
            '{"name": "support.get_order_status", "parameters": {"order_id": "10005"}}',
            '{"order_id": "10010", "user_id": "user_123"}',
            '{"tool": "get_order_status", "order_id": "10008"}',
            '[{"tool": "get_order_status"}]',
        ):
            assert is_customer_ready(leaked) is False, leaked

    def test_prose_is_still_an_answer(self) -> None:
        assert is_customer_ready("Order 10005 was delivered for $236.18.") is True

    def test_prose_containing_braces_is_still_an_answer(self) -> None:
        assert is_customer_ready("Your order {ref} shipped — see {link}.") is True

    async def test_the_customer_sees_prose_or_a_hand_off(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        llm = ScriptedLLM([{"content": '{"order_id": "10010", "user_id": "u1"}'}])
        agent = build_agent(llm=llm, pool=seeded_pool, with_memory=False)
        state = await agent.handle("what about 10010", session_id="s1")
        assert "order_id" not in state.response
        assert state.response


class TestPayloadLogging:
    def test_payload_logging_is_configurable(self) -> None:
        """The switch that shows what the model was actually asked."""
        assert isinstance(get_settings().llm.log_payloads, bool)

    def test_the_file_log_captures_more_than_the_console(self) -> None:
        """The file is the authoritative record, so it takes DEBUG."""
        assert get_settings().logging.file_level == "DEBUG"
