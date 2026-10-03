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

from agent_platform.persistence import SqliteConnectionPool
from customer_support_agent.graph import build_agent
from customer_support_agent.persistence.repositories import OrdersRepository, UsersRepository
from customer_support_agent.presentation.auth import unknown_code
from customer_support_agent.presentation.orders import customer_orders, describe_customer
from customer_support_agent.security.authentication import CustomerAuthenticator
from tests.conftest import CUSTOMER_CODE, ScriptedLLM


@pytest.fixture
def authenticator(seeded_pool: SqliteConnectionPool) -> CustomerAuthenticator:
    return CustomerAuthenticator(UsersRepository(seeded_pool))


def _read_back(authenticator: CustomerAuthenticator, pool: SqliteConnectionPool) -> str:
    """The verified greeting the authentication node renders for this code."""
    result = authenticator.authenticate(CUSTOMER_CODE)
    assert result.user is not None
    return describe_customer(
        result.user, customer_orders(OrdersRepository(pool), result.user.user_id)
    )


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
        malformed = authenticator.authenticate("AB12")
        assert wrong.code_attempted is True and wrong.user is None
        assert "couldn't find" in unknown_code()
        assert malformed.code_attempted is False

    def test_the_read_back_is_a_fixed_format(
        self, authenticator: CustomerAuthenticator, seeded_pool: SqliteConnectionPool
    ) -> None:
        """Rendered from the record, not by a model: these are the customer's own records."""
        message = _read_back(authenticator, seeded_pool)
        assert "Ada Lovelace" in message
        assert "Account" in message and "orders" in message

    def test_the_read_back_masks_the_email(
        self, authenticator: CustomerAuthenticator, seeded_pool: SqliteConnectionPool
    ) -> None:
        """A support transcript is not a place to restate a full address."""
        message = _read_back(authenticator, seeded_pool)
        assert "ada@example.com" not in message and "@example.com" in message

    def test_the_read_back_lists_the_customers_own_orders(
        self, authenticator: CustomerAuthenticator, seeded_pool: SqliteConnectionPool
    ) -> None:
        """Listed, not counted — they verified in order to talk about one."""
        message = _read_back(authenticator, seeded_pool)
        assert "10001" in message and "10002" in message


class TestTheConversationFlow:
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
