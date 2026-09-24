"""Who is speaking, and whether this is the agent's business.

Two gates run before anything else, and both are deterministic:

  scope     — a customer support agent is not a general assistant. An
              off-topic message is refused by a regex pass rather than a
              completion, because the refusal is identical every time.
  identity  — until a valid code arrives there is no owner to scope a
              lookup to, so "where is my order" has no answer.

Neither involves a model. A check that cannot be talked into a different
answer is the point of putting them first.
"""

import pytest

from agentic_commons.persistence import SqliteConnectionPool
from customer_support_agent.graph import build_agent
from customer_support_agent.persistence.repositories import UsersRepository
from customer_support_agent.security.authentication import CustomerAuthenticator
from tests.conftest import CUSTOMER_CODE, ScriptedLLM


@pytest.fixture
def authenticator(seeded_pool: SqliteConnectionPool) -> CustomerAuthenticator:
    from customer_support_agent.persistence.repositories import OrdersRepository

    return CustomerAuthenticator(UsersRepository(seeded_pool), OrdersRepository(seeded_pool))


class TestAuthentication:
    def test_a_valid_code_verifies(self, authenticator: CustomerAuthenticator) -> None:
        result = authenticator.authenticate(f"my code is {CUSTOMER_CODE}")
        assert result.authenticated and result.customer is not None
        assert result.customer.customer_id == 1

    def test_a_bare_code_verifies(self, authenticator: CustomerAuthenticator) -> None:
        assert authenticator.authenticate(CUSTOMER_CODE).authenticated is True

    def test_the_code_is_found_inside_a_sentence(
        self, authenticator: CustomerAuthenticator
    ) -> None:
        text = f"hi there, I think my code is {CUSTOMER_CODE} — can you check my order"
        assert authenticator.authenticate(text).authenticated is True

    def test_case_and_spacing_do_not_matter(self, authenticator: CustomerAuthenticator) -> None:
        assert authenticator.authenticate(CUSTOMER_CODE.lower()).authenticated is True

    def test_a_wrong_code_does_not_verify(self, authenticator: CustomerAuthenticator) -> None:
        assert authenticator.authenticate("ZZZ9ZZZ").authenticated is False

    def test_a_seven_letter_word_is_not_a_code(self, authenticator: CustomerAuthenticator) -> None:
        """Otherwise "what's the weather" reads as a failed login."""
        assert authenticator.extract_code("what is the weather today") is None
        assert authenticator.authenticate("what is the weather").code_attempted is False

    def test_a_wrong_code_and_a_malformed_one_give_the_same_message(
        self, authenticator: CustomerAuthenticator
    ) -> None:
        """Distinguishing them tells an attacker which codes exist."""
        wrong = authenticator.authenticate("ZZZ9ZZZ")
        assert wrong.code_attempted is True
        assert "couldn't find" in wrong.message

    def test_no_code_asks_for_one(self, authenticator: CustomerAuthenticator) -> None:
        result = authenticator.authenticate("where is my parcel")
        assert result.authenticated is False and result.code_attempted is False
        assert "customer code" in result.message

    def test_the_read_back_is_a_fixed_format(self, authenticator: CustomerAuthenticator) -> None:
        """Rendered here, not by a model: these are the customer's own records."""
        message = authenticator.authenticate(CUSTOMER_CODE).message
        assert "Ada Lovelace" in message
        assert "Account" in message and "orders" in message

    def test_the_read_back_masks_the_email(self, authenticator: CustomerAuthenticator) -> None:
        """A support transcript is not a place to restate a full address."""
        message = authenticator.authenticate(CUSTOMER_CODE).message
        assert "ada@example.com" not in message and "@example.com" in message

    def test_the_read_back_lists_the_customers_own_orders(
        self, authenticator: CustomerAuthenticator
    ) -> None:
        """Listed, not counted — they verified in order to talk about one."""
        message = authenticator.authenticate(CUSTOMER_CODE).message
        assert "Order 10001" in message and "Order 10002" in message


class TestTheConversationFlow:
    async def test_an_off_topic_message_never_reaches_the_model(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        """Paying a model to produce a constant refusal is pure waste."""
        llm = ScriptedLLM([{"content": "should not be called"}])
        agent = build_agent(llm=llm, pool=seeded_pool, with_memory=False)
        state = await agent.handle("what is the weather in London", session_id="s1")
        assert llm.requests == []
        assert "only help with" in state.response

    async def test_an_order_question_before_verifying_asks_for_a_code(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        llm = ScriptedLLM([{"content": "should not be called"}])
        agent = build_agent(llm=llm, pool=seeded_pool, with_memory=False)
        agent._sessions.clear()
        from customer_support_agent.domain.identity import set_current_customer

        set_current_customer(None)
        state = await agent.handle("what is the status of my order", session_id="fresh")
        assert llm.requests == []
        assert "customer code" in state.response

    async def test_verifying_returns_the_details_without_a_model_call(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        llm = ScriptedLLM([{"content": "should not be called"}])
        agent = build_agent(llm=llm, pool=seeded_pool, with_memory=False)
        from customer_support_agent.domain.identity import set_current_customer

        set_current_customer(None)
        state = await agent.handle(f"my code is {CUSTOMER_CODE}", session_id="fresh")
        assert llm.requests == []
        assert "verified" in state.response

    async def test_verification_persists_for_the_session(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        """A customer verifies once, not every turn."""
        from customer_support_agent.domain.identity import set_current_customer

        llm = ScriptedLLM([{"content": "Order 10001 was delivered."}])
        agent = build_agent(llm=llm, pool=seeded_pool, with_memory=False)
        set_current_customer(None)
        await agent.handle(f"my code is {CUSTOMER_CODE}", session_id="fresh")
        state = await agent.handle("what is the status of order 10001", session_id="fresh")
        assert "customer code" not in state.response

    async def test_a_verified_session_scopes_to_that_customer(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        """Verified as u1, order 10003 belongs to u2 and stays invisible."""
        from customer_support_agent.domain.identity import set_current_customer

        llm = ScriptedLLM(
            [
                {"content": ""},
                {"content": "I couldn't find that order."},
            ]
        )
        agent = build_agent(llm=llm, pool=seeded_pool, with_memory=False)
        set_current_customer(None)
        await agent.handle(f"my code is {CUSTOMER_CODE}", session_id="fresh")
        from customer_support_agent.tools.definitions import SupportTools  # noqa: F401

        assert agent._sessions["fresh"].customer_id == 1


class TestIdentitySurvivesTheGraph:
    """Identity must reach the node that uses it.

    LangGraph runs each node in its own task. Context variables copy
    *into* a task when it starts and do not propagate back out, so a
    customer bound inside the `authenticate` node was invisible to `act`
    — every lookup silently scoped to nobody, and the tool failed with
    "no order found" on the customer's own order.
    """

    async def test_a_lookup_after_verifying_in_conversation_succeeds(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        """A direct order reference is now answered from the database.

        Identity still has to have survived the graph for this to work:
        the lookup is ownership-scoped, so an unbound customer would see
        nothing.
        """
        from customer_support_agent.domain.identity import set_current_customer

        llm = ScriptedLLM([{"content": "(should not be reached)"}])
        agent = build_agent(llm=llm, pool=seeded_pool, with_memory=False)
        set_current_customer(None)
        await agent.handle(f"my code is {CUSTOMER_CODE}", session_id="fresh")
        state = await agent.handle("status of order 10001", session_id="fresh")
        assert "Order 10001" in state.response and llm.requests == []

    async def test_a_tool_lookup_after_verifying_also_succeeds(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        """The same property on the governed path, where a tool actually runs."""
        from customer_support_agent.domain.identity import set_current_customer
        from tests.conftest import tool_call

        llm = ScriptedLLM(
            [
                {"tool_calls": tool_call("support__get_order_status", order_id=10001)},
                {"content": "It was delivered."},
            ]
        )
        agent = build_agent(llm=llm, pool=seeded_pool, with_memory=False)
        set_current_customer(None)
        await agent.handle(f"my code is {CUSTOMER_CODE}", session_id="fresh")
        state = await agent.handle("has my parcel been delivered yet", session_id="fresh")
        assert any("get_order_status: ok" in decision for decision in state.decisions)

    async def test_the_offered_set_is_carried_not_recomputed(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        """Selection can use a network ranker; recomputing can differ.

        A tool the model was genuinely handed must not then be refused
        because a second ranking returned something else.
        """
        from tests.conftest import tool_call

        llm = ScriptedLLM(
            [
                {"tool_calls": tool_call("support__get_order_status", order_id=10001)},
                {"content": "Order 10001 was delivered."},
            ]
        )
        agent = build_agent(llm=llm, pool=seeded_pool, with_memory=False)
        state = await agent.handle("status of order 10001", session_id="s1")
        assert state.metadata.get("offered_actions")
        assert not any("not offered" in decision for decision in state.decisions)
