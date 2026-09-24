"""Who is asking, and which order they mean.

Nothing downstream may act on a customer's behalf until this has settled
who the customer is. Identity is established deterministically, from the
database — never from something the model concluded.
"""

import re
from typing import Any

from customer_support_agent.domain.identity import (
    current_customer,
    set_current_customer,
)
from customer_support_agent.graph.nodes._support import NodeSupport
from customer_support_agent.graph.state import AgentState
from customer_support_agent.presentation.orders import (
    build_menu,
    customer_orders,
    render_order_detail,
    render_order_list,
    resolve_selection,
)
from customer_support_agent.presentation.reporting import (
    asks_for_the_list,
    extract_order_id,
)


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
                "response": result.message,
                "decisions": [
                    *state.decisions,
                    "auth: rejected" if result.code_attempted else "auth: awaiting code",
                ],
                "metadata": {**state.metadata, "halted": "unauthenticated"},
            }
        assert result.customer is not None
        if state.session_id:
            self._sessions[state.session_id] = result.customer
            if result.menu is not None:
                self._menus[state.session_id] = result.menu
        set_current_customer(result.customer)
        self._done(
            "authentication",
            summary="Customer verified",
            fields={
                "verified": True,
                "method": "customer_code",
                "customer_id": result.customer.customer_id,
                "orders": result.menu.size if result.menu is not None else 0,
            },
        )
        self._note(
            state,
            "verified_customer",
            f"{result.customer.name} (account #{result.customer.customer_id})",
            category="identity",
        )
        return {
            "response": result.message,
            "customer_id": result.customer.customer_id,
            "decisions": [*state.decisions, f"auth: verified #{result.customer.customer_id}"],
            # The verified read-back *is* the answer to "here is my code".
            # Nothing further is needed this turn.
            "metadata": {**state.metadata, "halted": "just_verified"},
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
        """
        menu = self._menus.get(state.session_id)
        if menu is None or self._orders is None or state.customer_id is None:
            return {}
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
            return self._halt(state, render_order_list(orders), "menu: listed")
        order_id = resolve_selection(state.query, menu)
        if order_id is None:
            return {}
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
                "amount": f"${order.amount:,.2f}",
                "found": True,
            },
        )
        self._note(
            state,
            "order_under_discussion",
            f"order {order.order_id}, {order.order_status}, ${order.amount:,.2f}",
            category="order",
        )
        return self._halt(state, render_order_detail(order), f"menu: order {order_id}")

    def _resolve_read_order_id(self, state: AgentState) -> int | None:
        """Resolve the most recent order under discussion, newest evidence first."""
        metadata_id = state.metadata.get("order_under_discussion")
        if isinstance(metadata_id, int) and metadata_id > 0:
            return metadata_id

        explicit = extract_order_id(state.query)
        if explicit is not None:
            return explicit

        # Working memory is recency ordered. The first matching fact is the
        # most recent validated order reference; do not search older facts
        # once one is found. This keeps follow-ups anchored to the latest
        # order without asking the model to infer which identifier wins.
        if self._memory is not None and state.session_id:
            for fact in self._memory.active_window(state.session_id):
                if fact.key != "order_under_discussion":
                    continue
                match = re.search(r"\border\s+(\d+)\b", fact.value, flags=re.IGNORECASE)
                if match:
                    return int(match.group(1))
        return None
