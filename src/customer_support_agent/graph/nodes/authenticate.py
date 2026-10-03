"""Who is asking, and which order they mean.

Nothing downstream may act on a customer's behalf until this has settled
who the customer is. Identity is established deterministically, from the
database — never from something the model concluded.
"""

from typing import Any

from customer_support_agent.domain.identity import (
    current_customer,
    set_current_customer,
)
from customer_support_agent.domain.order_reference import ORDER_UNDER_DISCUSSION
from customer_support_agent.graph.nodes._support import NodeSupport
from customer_support_agent.graph.state import AgentState
from customer_support_agent.presentation.auth import ask_for_code, unknown_code
from customer_support_agent.presentation.orders import (
    build_menu,
    customer_orders,
    describe_customer,
    render_order_detail,
    render_order_list,
    resolve_selection,
    summarise_order_list,
    summarise_verification,
)
from customer_support_agent.presentation.reporting import asks_for_the_list

# The turn that verified a customer. It carries their code and nothing they
# asserted, so long-term memory has no fact to take from it.
JUST_VERIFIED = "just_verified"


class AuthenticationNodes(NodeSupport):
    async def authenticate(self, state: AgentState) -> dict[str, Any]:
        """Establish who is speaking before looking anything up.

        Until a valid code arrives there is no owner to scope a lookup
        to, so "where is my order" has no answer. Verification is a
        database lookup, not a model call: the code either matches a row
        or it does not, and a deterministic check cannot be talked into a
        different answer.
        """
        existing = self._sessions.get(state.session_id) or current_customer()
        self._report(
            "authentication",
            summary="Verifying the customer",
        )
        if existing is not None:
            # Already verified — either earlier in this conversation, or
            # by the caller before it handed the request over.
            set_current_customer(existing)
            if state.session_id:
                self._sessions[state.session_id] = existing
            self._done(
                "authentication",
                summary="Customer already verified",
                fields={
                    "verified": True,
                    "method": "session",
                    "customer_id": existing.customer_id,
                },
            )
            return {"customer_id": existing.customer_id}
        if self._authenticator is None:
            self._done(
                "authentication",
                summary="Authentication delegated upstream",
                fields={"verified": False, "method": "upstream"},
            )
            return {}
        result = self._authenticator.authenticate(state.query)
        # Whatever happens next, the code typed this turn is not kept: history,
        # the episodic record and any later prompt read `model_query`.
        masked = (
            {"model_query": self._authenticator.mask_code(state.query)}
            if result.code_attempted
            else {}
        )
        if not result.authenticated:
            if result.code_attempted:
                self._reporter_blocked(
                    "authentication",
                    summary="Customer code was not accepted",
                    fields={
                        "verified": False,
                        "method": "customer_code",
                        "reason": "invalid_or_unknown_code",
                    },
                )
            else:
                self._reporter_waiting(
                    "authentication",
                    summary="Waiting for customer verification",
                    fields={
                        "verified": False,
                        "method": "customer_code",
                        "action": "request_code",
                    },
                )
            return {
                **masked,
                "response": unknown_code() if result.code_attempted else ask_for_code(),
                "decisions": [
                    *state.decisions,
                    "auth: rejected" if result.code_attempted else "auth: awaiting code",
                ],
                "metadata": {**state.metadata, "halted": "unauthenticated"},
            }
        assert result.customer is not None and result.user is not None
        # The read-back lists the customer's orders: they verified in order
        # to talk about one, and the list is the menu a bare "2" resolves in.
        orders = (
            customer_orders(self._orders, result.customer.customer_id)
            if self._orders is not None
            else []
        )
        menu = build_menu(orders)
        if state.session_id:
            self._sessions[state.session_id] = result.customer
            self._menus[state.session_id] = menu
        set_current_customer(result.customer)
        self._done(
            "authentication",
            summary="Customer verified",
            fields={
                "verified": True,
                "method": "customer_code",
                "customer_id": result.customer.customer_id,
                "orders": menu.size,
            },
        )
        self._note(
            state,
            "verified_customer",
            f"{result.customer.name} (account #{result.customer.customer_id})",
            category="identity",
        )
        return {
            **masked,
            "response": describe_customer(result.user, orders),
            "customer_id": result.customer.customer_id,
            "decisions": [*state.decisions, f"auth: verified #{result.customer.customer_id}"],
            # The verified read-back *is* the answer to "here is my code".
            # Nothing further is needed this turn.
            "metadata": {
                **state.metadata,
                "halted": JUST_VERIFIED,
                # The greeting carries the customer's whole order list.
                # On screen that saves them a round trip; replayed into
                # every later prompt it is the single largest thing the
                # model reads, and about none of what it is asked.
                "memory_response": summarise_verification(result.customer.name, menu.size),
            },
        }

    async def choose_order(self, state: AgentState) -> dict[str, Any]:
        """Answer a menu choice straight from the database.

        "2" after being shown a list is a lookup, not a conversation.
        Resolving it here costs one SELECT; routing it through the model
        costs a completion to read a row aloud, and risks the model
        getting a digit wrong about the customer's own order.

        Only an explicit menu-number reply lands here.
        "order 10560" is an order reference, not a menu command, and
        continues through the normal conversational path.

        This node also settles which order is under discussion for the
        rest of the turn, because it runs before every stage that can be
        switched off — a reference resolved here is available whatever
        this agent's harness is set to.
        """
        carried = self._carry_order_reference(state)
        if self._orders is None or state.customer_id is None:
            return carried
        # The list is answered from the record whether or not one was shown
        # before: a customer verified upstream has no menu yet, and "show
        # orders" sent to the model came back as prose about their orders.
        if asks_for_the_list(state.query):
            orders = customer_orders(self._orders, state.customer_id)
            self._menus[state.session_id] = build_menu(orders)
            self._report(
                "order_lookup",
                summary="Loading the customer's orders",
            )
            self._done(
                "order_lookup",
                summary="Order list loaded",
                fields={"count": len(orders)},
            )
            return self._halt(
                state,
                render_order_list(orders),
                "menu: listed",
                remembered_as=summarise_order_list(orders),
            )
        menu = self._menus.get(state.session_id)
        order_id = resolve_selection(state.query, menu) if menu is not None else None
        if order_id is None:
            return carried
        self._report(
            "order_lookup",
            summary="Checking order details",
            fields={"order_id": order_id},
        )
        order = self._orders.get_for_customer(order_id, state.customer_id)
        if order is None:
            self._reporter_blocked(
                "order_lookup",
                summary="Order not found on this account",
                fields={"order_id": order_id, "found": False},
            )
            return self._halt(
                state, f"I couldn't find order {order_id} on your account.", "menu: not found"
            )
        self._done(
            "order_lookup",
            summary="Order found",
            fields={
                "order_id": order.order_id,
                "status": order.order_status,
                "refund_status": order.refund_status,
                "amount": str(order.amount),
                "found": True,
            },
        )
        self._note(
            state,
            ORDER_UNDER_DISCUSSION,
            f"order {order.order_id}, {order.order_status}, {order.amount}",
            category="order",
        )
        return self._halt(state, render_order_detail(order), f"menu: order {order_id}")
