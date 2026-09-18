"""Fixed-format views rendered from the database (LLD §21).

Everything here is a straight query and a formatter. No model is
involved, deliberately.

These are the customer's own records — their name, their orders, their
amounts. A model asked to render them can get a digit wrong, drop an
order, or invent a status, and being subtly wrong about somebody's
account is worse than being terse. A query cannot do any of those.

It is also the cheap path: listing a customer's orders is one SELECT, and
paying for a completion to read a table aloud is waste.
"""

import re

from pydantic import BaseModel

from customer_support_agent.core.repositories import OrdersRepository
from customer_support_agent.core.schema import Order, User

# Words a customer wraps a menu choice in. Stripped before deciding
# whether what remains is *only* a selection — "2" and "order 2" are
# choices, "refund order 2" is a request and belongs on the normal path.
_SELECTION_FILLER = re.compile(
    r"\b(order|orders|number|no|num|option|item|show|see|me|my|the|details?|detail|"
    r"about|for|on|please|tell|more|info|information|check|view|pick|choose|"
    r"what|whats|how|and|also|then|next|status|of)\b",
    re.IGNORECASE,
)
_SELECTION_NOISE = re.compile(r"[^\w\s]")

# How an order's state reads to a customer.
_STATUS_WORDS = {
    "pending": "being prepared",
    "shipped": "on its way",
    "delivered": "delivered",
    "cancelled": "cancelled",
}
_REFUND_WORDS = {
    "none": "",
    "pending_review": "refund awaiting review",
    "approved": "refund approved",
    "denied": "refund declined",
    "completed": "refunded in full",
}


class OrderMenu(BaseModel):
    """The numbered list of orders a customer was last shown.

    Kept per session so a bare "2" can be resolved back to an order. The
    menu position is what the customer sees; the order id is what the
    system uses, and both resolve.
    """

    order_ids: list[int]

    def at(self, position: int) -> int | None:
        """The order id at a 1-based menu position."""
        if 1 <= position <= len(self.order_ids):
            return self.order_ids[position - 1]
        return None

    def contains(self, order_id: int) -> bool:
        return order_id in self.order_ids

    @property
    def size(self) -> int:
        return len(self.order_ids)


def describe_customer(user: User, orders: list[Order]) -> str:
    """The verification read-back: who they are, and what they have.

    The order list is part of the greeting rather than something the
    customer has to ask for. They authenticated in order to talk about an
    order; making them request the list first is a round trip that
    answers a question nobody had.
    """
    lines = [
        f"Thanks, {user.name} — you're verified.",
        f"  Account        : #{user.user_id}",
        f"  Email          : {user.masked_email}",
        f"  Customer since : {user.created_at[:10]}",
        "",
    ]
    if not orders:
        lines.append("You have no orders on this account yet.")
        return "\n".join(lines)

    lines.append(f"Your {len(orders)} order{'s' if len(orders) != 1 else ''}:")
    lines.append("")
    lines.extend(_menu_lines(orders))
    lines.append("")
    lines.append("Reply with a number to see an order, or tell me what you need.")
    return "\n".join(lines)


def render_order_list(orders: list[Order]) -> str:
    """The numbered list on its own, for a customer who asks again."""
    if not orders:
        return "You have no orders on this account."
    return "\n".join([f"Your {len(orders)} orders:", "", *_menu_lines(orders)])


def render_order_detail(order: Order, include_refund: bool = False) -> str:
    """What is known about one order.

    Refund state is omitted unless it was asked for. A customer checking
    where their parcel is has not raised a refund, and answering with one
    invites a request nobody intended to make — the same reason the
    status tool does not report it either.
    """
    lines = [
        f"Order {order.order_id}",
        f"  Placed : {order.timestamp[:10]}",
        f"  Status : {_STATUS_WORDS.get(order.order_status, order.order_status)}",
        f"  Total  : ${order.amount:,.2f}",
    ]
    if include_refund:
        refund = _REFUND_WORDS.get(order.refund_status, order.refund_status)
        lines.append(f"  Refund : {refund or 'none requested'}")
    return "\n".join(lines)


def build_menu(orders: list[Order]) -> OrderMenu:
    """The menu backing the list a customer was just shown."""
    return OrderMenu(order_ids=[order.order_id for order in orders])


def resolve_selection(message: str, menu: OrderMenu) -> int | None:
    """The order id a message selects, or None if it is not a selection.

    Only a message that is *nothing but* a choice resolves. "2",
    "order 2" and "what about 10286" pick from the menu, because all
    three are asking to look at something. "refund order 2" states an
    intent and belongs on the normal path, where the Gatekeeper and the
    refund rules apply — hijacking it here would route a refund request
    through a read-only view and answer the wrong question.

    The filler list is what draws that line: words that only frame a
    request are stripped, and any word left over means the customer
    asked for something more than a look.
    """
    stripped = _SELECTION_NOISE.sub(" ", message)
    remainder = _SELECTION_FILLER.sub(" ", stripped)
    tokens = remainder.split()
    if len(tokens) != 1 or not tokens[0].isdigit():
        return None
    value = int(tokens[0])
    # A menu position, or the order id itself — the customer may read
    # either off the list.
    if menu.contains(value):
        return value
    return menu.at(value)


def customer_orders(orders_repository: OrdersRepository, customer_id: int) -> list[Order]:
    """A customer's orders, newest first."""
    found = orders_repository.for_user(customer_id)
    return sorted(found, key=lambda order: order.timestamp, reverse=True)


def _menu_lines(orders: list[Order]) -> list[str]:
    """One aligned row per order: position, id, date, status, total.

    No refund column. Listing orders is not a refund conversation, and a
    list that annotates every row with refund state is advertising.
    """
    return [
        f"  {position:>2}. Order {order.order_id}  {order.timestamp[:10]}  "
        f"{_STATUS_WORDS.get(order.order_status, order.order_status):<14} "
        f"${order.amount:>9,.2f}"
        for position, order in enumerate(orders, start=1)
    ]
