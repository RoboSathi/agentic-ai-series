"""Verification lists the customer's orders, and a choice is answered from the DB.

Everything here is a query and a formatter. No model is involved, which
is both the cheap path and the correct one: these are the customer's own
records, and a model asked to read a table aloud can drop an order or get
a digit wrong.

The boundary that matters is what counts as a *selection*. "2" and "what
about 10286" are asking to look at something; "refund order 2" states an
intent and must continue to the governed path where the Gatekeeper and
the refund rules apply.
"""

import pytest

from agentic_commons.llm.types import CompletionResult, HealthCheckResult, TokenUsage
from agentic_commons.persistence import SqliteConnectionPool
from customer_support_agent.domain.identity import set_current_customer
from customer_support_agent.graph import build_agent
from customer_support_agent.persistence import OrdersRepository
from customer_support_agent.persistence.repositories import UsersRepository
from customer_support_agent.presentation.orders import (
    OrderMenu,
    build_menu,
    customer_orders,
    describe_customer,
    render_order_detail,
    render_order_list,
    resolve_selection,
)
from customer_support_agent.security.authentication import CustomerAuthenticator
from tests.conftest import CUSTOMER_CODE, ScriptedLLM


class CountingLLM(ScriptedLLM):
    """Records whether the model was reached at all."""

    def __init__(self) -> None:
        super().__init__([{"content": "(the model was called)"}])
        self.calls = 0

    async def complete(self, request):  # type: ignore[no-untyped-def]
        self.calls += 1
        return CompletionResult(content="(the model was called)", usage=TokenUsage(total_tokens=10))

    async def health_check(self) -> HealthCheckResult:
        return HealthCheckResult(healthy=True)


@pytest.fixture
def menu() -> OrderMenu:
    return OrderMenu(order_ids=[10001, 10002, 10003])


class TestSelectionBoundary:
    @pytest.mark.parametrize(
        "message",
        [
            "refund order 2",
            "I want a refund for 10002",
            "how do I refund 10002",
            "cancel 2",
            "who are you",
            "",
        ],
    )
    def test_a_request_does_not(self, message: str, menu: OrderMenu) -> None:
        """Anything more than a look belongs on the governed path."""
        assert resolve_selection(message, menu) is None

    def test_a_position_outside_the_menu_resolves_to_nothing(self, menu: OrderMenu) -> None:
        assert resolve_selection("99", menu) is None

    def test_a_position_and_an_order_id_both_work(self, menu: OrderMenu) -> None:
        """A customer may read either off the list."""
        assert resolve_selection("1", menu) == resolve_selection("10001", menu)


class TestRendering:
    def test_the_list_numbers_every_order(self, seeded_pool: SqliteConnectionPool) -> None:
        orders = customer_orders(OrdersRepository(seeded_pool), 1)
        rendered = render_order_list(orders)
        assert all(f"Order {order.order_id}" in rendered for order in orders)
        assert "1." in rendered and "2." in rendered

    def test_the_list_shows_status_in_plain_words(self, seeded_pool: SqliteConnectionPool) -> None:
        rendered = render_order_list(customer_orders(OrdersRepository(seeded_pool), 1))
        assert "delivered" in rendered

    def test_an_empty_account_says_so(self) -> None:
        assert "no orders" in render_order_list([])

    def test_detail_includes_the_amount_and_status(self, seeded_pool: SqliteConnectionPool) -> None:
        order = OrdersRepository(seeded_pool).get(10001)
        assert order is not None
        detail = render_order_detail(order)
        assert "10001" in detail and "32.00" in detail and "delivered" in detail

    def test_detail_stays_silent_about_refunds_until_asked(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        """A customer checking where a parcel is has not raised a refund.

        Volunteering refund state on every order view invites a request
        nobody intended to make, which is how an agent ends up issuing
        money back on an enquiry.
        """
        order = OrdersRepository(seeded_pool).get(10001)
        assert order is not None
        assert "refund" not in render_order_detail(order).casefold()
        assert "refund" in render_order_detail(order, include_refund=True).casefold()

    def test_the_greeting_lists_the_orders(self, seeded_pool: SqliteConnectionPool) -> None:
        user = UsersRepository(seeded_pool).get(1)
        orders = customer_orders(OrdersRepository(seeded_pool), 1)
        assert user is not None
        greeting = describe_customer(user, orders)
        assert "you're verified" in greeting
        assert f"Order {orders[0].order_id}" in greeting
        assert "Reply with a number" in greeting

    def test_the_greeting_masks_the_email(self, seeded_pool: SqliteConnectionPool) -> None:
        user = UsersRepository(seeded_pool).get(1)
        assert user is not None
        assert "ada@example.com" not in describe_customer(user, [])

    def test_orders_are_listed_newest_first(self, seeded_pool: SqliteConnectionPool) -> None:
        orders = customer_orders(OrdersRepository(seeded_pool), 1)
        assert [order.order_id for order in orders] == [10002, 10001]


class TestTheConversationNeedsNoModel:
    @pytest.fixture
    def agent(self, seeded_pool: SqliteConnectionPool):  # type: ignore[no-untyped-def]
        llm = CountingLLM()
        built = build_agent(llm=llm, pool=seeded_pool, with_memory=False)
        set_current_customer(None)
        return built, llm

    async def test_verifying_lists_the_orders(self, agent) -> None:  # type: ignore[no-untyped-def]
        built, llm = agent
        state = await built.handle(f"my code is {CUSTOMER_CODE}", session_id="s")
        assert "Order 10001" in state.response and "Order 10002" in state.response
        assert llm.calls == 0

    async def test_choosing_a_number_answers_from_the_database(self, agent) -> None:  # type: ignore[no-untyped-def]
        built, llm = agent
        await built.handle(f"my code is {CUSTOMER_CODE}", session_id="s")
        state = await built.handle("1", session_id="s")
        assert "Order 10002" in state.response
        assert llm.calls == 0

    async def test_asking_for_the_list_again_needs_no_model(self, agent) -> None:  # type: ignore[no-untyped-def]
        built, llm = agent
        await built.handle(f"my code is {CUSTOMER_CODE}", session_id="s")
        state = await built.handle("show me my orders", session_id="s")
        assert "Order 10001" in state.response
        assert llm.calls == 0

    async def test_the_menu_only_reaches_the_customers_own_orders(
        self, agent, seeded_pool: SqliteConnectionPool
    ) -> None:  # type: ignore[no-untyped-def]
        """Order 10003 belongs to u2; selecting its id must not disclose it."""
        built, llm = agent
        await built.handle(f"my code is {CUSTOMER_CODE}", session_id="s")
        state = await built.handle("10003", session_id="s")
        assert "$45.00" not in state.response


class TestTheAuthenticatorSuppliesTheMenu:
    def test_authentication_returns_a_menu(self, seeded_pool: SqliteConnectionPool) -> None:
        authenticator = CustomerAuthenticator(
            UsersRepository(seeded_pool), OrdersRepository(seeded_pool)
        )
        result = authenticator.authenticate(CUSTOMER_CODE)
        assert result.menu is not None and result.menu.size == 2

    def test_a_failed_attempt_carries_no_menu(self, seeded_pool: SqliteConnectionPool) -> None:
        authenticator = CustomerAuthenticator(
            UsersRepository(seeded_pool), OrdersRepository(seeded_pool)
        )
        assert authenticator.authenticate("ZZZ9ZZZ").menu is None

    def test_the_menu_matches_what_was_shown(self, seeded_pool: SqliteConnectionPool) -> None:
        orders = customer_orders(OrdersRepository(seeded_pool), 1)
        assert build_menu(orders).order_ids == [order.order_id for order in orders]
