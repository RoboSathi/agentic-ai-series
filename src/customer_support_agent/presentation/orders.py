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

from pydantic import BaseModel

from customer_support_agent.domain.models import Order, User
from customer_support_agent.persistence.repositories import OrdersRepository

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
        f"  Item   : {order.description}",
        f"  Placed : {order.timestamp[:10]}",
        f"  Status : {_STATUS_WORDS.get(order.order_status, order.order_status)}",
    ]
    if order.quantity > 1:
        lines.append(f"  Each   : ${order.unit_price:,.2f}")
    lines.append(f"  Total  : ${order.amount:,.2f}")
    # Only when there is something true to say. A tracking line on an
    # order nobody has shipped reads as a promise the record cannot keep.
    if order.is_trackable or order.estimated_delivery_date:
        lines.append(f"  Parcel : {order.whereabouts}")
    if order.tracking_number and order.carrier:
        lines.append(f"  Track  : {order.carrier} {order.tracking_number}")
    if include_refund:
        refund = _REFUND_WORDS.get(order.refund_status, order.refund_status)
        lines.append(f"  Refund : {refund or 'none requested'}")
    return "\n".join(lines)


def build_menu(orders: list[Order]) -> OrderMenu:
    """The menu backing the list a customer was just shown."""
    return OrderMenu(order_ids=[order.order_id for order in orders])


def resolve_selection(message: str, menu: OrderMenu) -> int | None:
    """Resolve only an explicit menu-number reply.

    Menu interaction is deliberately separate from conversation. A bare
    number is the UI gesture described by the menu ("Reply with a number"),
    while text such as "order 10560" is a conversational order reference and
    must continue through BasicSanity → Intent → the normal agent path.

    This distinction matters when the assistant has just asked which order
    should be refunded: showing order details would answer a different
    question and would bypass the user's pending intent.
    """
    normalized = message.strip()
    if not normalized.isdigit():
        return None
    value = int(normalized)
    if menu.contains(value):
        return value
    return menu.at(value)


def customer_orders(orders_repository: OrdersRepository, customer_id: int) -> list[Order]:
    """A customer's orders, newest first."""
    found = orders_repository.for_user(customer_id)
    return sorted(found, key=lambda order: order.timestamp, reverse=True)


# How much of a product name one menu row shows before it is elided.
_ITEM_WIDTH = 34


def _menu_lines(orders: list[Order]) -> list[str]:
    """One aligned row per order: id, date, order status, refund status, total.

    The list is rendered deterministically so the customer sees the same
    authoritative refund state that the database returned. This function is
    shared by the post-authentication greeting and an explicit "show orders"
    request, so both surfaces stay consistent.
    """
    return [
        f"  {position:>2}. Order {order.order_id}  {order.timestamp[:10]}  "
        f"{_shorten(order.description):<34} "
        f"{_STATUS_WORDS.get(order.order_status, order.order_status):<14} "
        f"{('Refund: ' + _REFUND_WORDS.get(order.refund_status, order.refund_status) if _REFUND_WORDS.get(order.refund_status, order.refund_status) else 'Refund: none'): <25} "
        f"${order.amount:>9,.2f}"
        for position, order in enumerate(orders, start=1)
    ]


def _shorten(text: str) -> str:
    """Keep one order to one line, whatever it is called."""
    return text if len(text) <= _ITEM_WIDTH else text[: _ITEM_WIDTH - 1] + "…"
