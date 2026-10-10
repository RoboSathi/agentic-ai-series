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

from agent_platform.context.types import Chunk
from agent_platform.presentation import Align, box_table
from customer_support_agent.config import get_agent_settings
from customer_support_agent.config.settings import OrderDetailPhrases
from customer_support_agent.domain.models import Order, User
from customer_support_agent.persistence.repositories import OrdersRepository


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


# How a record states an order's refund state. The hallucination guard's
# state check reads this line (`record_state_pattern`) to judge any claim
# about where a refund stands, so every record of an order ends with it.
REFUND_STATUS_LINE = "Order {order_id} refund status: {status}"

# A waiting review, as its own sentence so it is its own premise. The earlier
# "(a review request is already pending)" read to NLI as *contradicting* "a
# review request has been submitted" — pending and submitted are the same
# fact in two words, and a parenthesis gave it nothing to hold on to.
_REVIEW_PENDING = (
    "A refund review request for order {order_id} has been submitted and is awaiting a decision"
)
# Each refund state in words. With REFUND_STATUS answered straight from this
# tool, the sentence is the customer's reply; the status line after it is
# what the guard's deterministic refund-state check reads. No arrival date is
# promised: the record holds none.
_REFUND_STATE_SENTENCES: dict[str, str] = {
    "none": "No refund has been requested for order {order_id}",
    "pending_review": _REVIEW_PENDING,
    "issued": (
        "The refund of {amount} for order {order_id} has been issued and is waiting to settle"
    ),
    "completed": "The refund of {amount} for order {order_id} has been completed",
    "denied": "The refund request for order {order_id} was declined by a reviewer",
}


def refund_state_sentence(order: Order) -> str:
    """Where this order's refund stands, as a sentence a customer can read."""
    template = _REFUND_STATE_SENTENCES.get(order.refund_status, REFUND_STATUS_LINE)
    return template.format(order_id=order.order_id, amount=order.amount, status=order.refund_status)


# The order under discussion, read fresh from the database for a reply to be
# grounded on. It ranks with tool results (the guard's record tier) and its
# source says a record, not a tool, established it.
ORDER_RECORD_SOURCE = "order_record:{order_id}"


def order_sentence(order: Order) -> str:
    """One order as a sentence: what context shows the model, and what a record states."""
    return (
        f"{order_state_sentence(order, placed=True)}, {order.amount}, refund {order.refund_status}"
    )


def order_state_sentence(order: Order, placed: bool = False) -> str:
    """Which order this is and where it has got to, as a sentence NLI can read.

    The grounding check compares a claim with one sentence at a time, and a
    comma list of status words is not one: "Order N: item, shipped, in
    transit" read as *contradicting* "your order has already shipped".
    """
    wording = get_agent_settings().order_sentences
    return f"{_order_subject(order, placed)} {_order_state(order, wording.details)}"


def _order_subject(order: Order, placed: bool) -> str:
    """The order by number, with what was bought when the record names it."""
    wording = get_agent_settings().order_sentences
    template = wording.subject_with_item if order.product_name else wording.subject
    subject = template.format(order_id=order.order_id, description=order.description)
    return (
        wording.placed.format(subject=subject, placed=order.timestamp[:10]) if placed else subject
    )


def _order_state(order: Order, details: OrderDetailPhrases) -> str:
    """The state clause for this order's status, with only the details the record holds."""
    states = get_agent_settings().order_sentences.states.model_dump()
    return str(states[order.order_status]).format(
        location=_detail(details.location, order.current_location),
        due=_detail(details.due, (order.estimated_delivery_date or "")[:10]),
        delivered=_detail(details.delivered, (order.delivered_at or "")[:10]),
    )


def _detail(phrase: str, value: str | None) -> str:
    """One optional phrase, or nothing when the record holds no value for it."""
    return phrase.format(value=value) if value else ""


def order_record(orders: OrdersRepository, order_id: int, customer_id: int | None) -> Chunk | None:
    """The order as the database holds it now, if this customer owns it.

    Read, never parsed out of the prompt window: what the model was shown
    can be crowded or tampered with, the row cannot. It ends with the
    refund-status line, so the guard's state check judges any claim about
    where the refund stands against it.
    """
    if customer_id is None:
        return None
    order = orders.get_for_customer(order_id, customer_id)
    if order is None:
        return None
    status = REFUND_STATUS_LINE.format(order_id=order.order_id, status=order.refund_status)
    return Chunk(
        content=f"{order_sentence(order)}. {status}",
        kind="tool_result",
        source=ORDER_RECORD_SOURCE.format(order_id=order.order_id),
    )


def describe_orders(orders: list[Order]) -> str:
    """One plain sentence per order, as a tool states what it read.

    The menu table is for eyes; this is for a reply to be checked against.
    A column-padded row is a poor premise for the grounding check, which
    reads one line at a time and needs each to say what it means.
    """
    if not orders:
        return render_order_list(orders)
    return "\n".join(
        f"{order_state_sentence(order, placed=True)}, refund {order.refund_status}, "
        f"total {order.amount}"
        for order in orders
    )


def summarise_verification(name: str, order_count: int) -> str:
    """The verification greeting as working memory should keep it."""
    return f"Verified {name} and listed their {order_count} orders."


def summarise_order_list(orders: list[Order]) -> str:
    """The order list as working memory should keep it.

    The rendered table is forty lines wide and is the right answer on
    screen. Replayed into every subsequent prompt it is forty lines of
    the model re-reading orders nobody has mentioned since — which is
    how a fifteen-word refund request came to cost three thousand tokens.
    What the model actually needs carried forward is that the list was
    shown and how long it was; the orders themselves are one scoped
    repository read away on the turn that names one.
    """
    if not orders:
        return "Told the customer they have no orders on this account."
    return f"Listed the customer's {len(orders)} orders for them to choose from."


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
        f"  Status : {_status_word(order)}",
    ]
    if order.quantity > 1:
        lines.append(f"  Each   : {order.unit_price}")
    lines.append(f"  Total  : {order.amount}")
    # Only when there is something true to say. A tracking line on an
    # order nobody has shipped reads as a promise the record cannot keep.
    if order.is_trackable or order.estimated_delivery_date:
        lines.append(f"  Parcel : {order.whereabouts}")
    if order.tracking_number and order.carrier:
        lines.append(f"  Track  : {order.carrier} {order.tracking_number}")
    if include_refund:
        words = get_agent_settings().order_table
        lines.append(
            f"  Refund : {words.refund_words.get(order.refund_status) or words.no_refund_detail}"
        )
    return "\n".join(lines)


def build_menu(orders: list[Order]) -> OrderMenu:
    """The menu backing the list a customer was just shown."""
    return OrderMenu(order_ids=[order.order_id for order in orders])


def resolve_selection(message: str, menu: OrderMenu) -> int | None:
    """Resolve only an explicit menu-number reply.

    Menu interaction is deliberately separate from conversation. A bare
    number is the UI gesture described by the menu ("Reply with a number"),
    while text such as "order 10560" is a conversational order reference and
    must continue through DeterministicCheck → Intent → the normal agent path.

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


# Column alignment: numbers and money right, words left.
_ALIGN: tuple[Align, ...] = ("right", "left", "left", "left", "left", "left", "right")


def _menu_lines(orders: list[Order]) -> list[str]:
    """One box table, one row per order: position, id, date, item, status, refund, total.

    Rendered from the record so the customer sees the same authoritative
    refund state the database returned. Shared by the verified greeting and
    an explicit "show orders", so both read the same.
    """
    table = get_agent_settings().order_table
    rows = [
        [
            str(position),
            str(order.order_id),
            order.timestamp[:10],
            _shorten(order.description),
            _status_word(order),
            table.refund_words.get(order.refund_status) or table.no_refund,
            str(order.amount),
        ]
        for position, order in enumerate(orders, start=1)
    ]
    return box_table(table.headers, rows, _ALIGN)


def _status_word(order: Order) -> str:
    """An order's state in the customer's words (configured), else the raw status."""
    return get_agent_settings().order_table.status_words.get(order.order_status, order.order_status)


def _shorten(text: str) -> str:
    """Keep one order to one line, whatever it is called."""
    table = get_agent_settings().order_table
    if len(text) <= table.item_width:
        return text
    return text[: table.item_width - len(table.ellipsis)] + table.ellipsis
