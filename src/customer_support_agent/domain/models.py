"""What this business is made of (LLD §21).

The domain vocabulary is deliberately narrow: an Order, a User, and a
request waiting on a human. Everything else — authority, memory,
context, resilience — belongs to `agent_platform` and arrives by
injection.

Nothing here reaches a database or a model. These are the nouns; the
verbs live in `tools/`, the tables in `persistence/`, and the state of a
turn in `graph/state.py`.
"""

from typing import Literal

from pydantic import BaseModel, Field

from agent_platform.observability.types import utc_now
from agent_platform.primitives.money import Money

OrderStatus = Literal["pending", "shipped", "delivered", "cancelled"]
RefundStatus = Literal["none", "pending_review", "approved", "denied", "completed"]
HitlStatus = Literal["pending", "approved", "denied"]
# What a reviewer's approval will do. A cancellation that moves money is
# reviewed as one decision, so approving it cancels *and* refunds.
ReviewAction = Literal["refund", "cancel_and_refund"]

# An order is refundable only once it is settled. Domain invariant, not a
# tunable: it is what "refund" means for this business, and the dummy
# data generator applies the same rule so the fixtures cannot contradict
# the policy the agent is asked to enforce.
REFUNDABLE_ORDER_STATUSES: frozenset[str] = frozenset({"delivered", "cancelled"})
# An order can be cancelled only before it leaves the warehouse. Once it
# has shipped, stopping it is a carrier recall this business does not do.
CANCELLABLE_ORDER_STATUSES: frozenset[str] = frozenset({"pending"})


class Order(BaseModel):
    """One customer order: what was bought, and where it has got to."""

    order_id: int
    user_id: int
    timestamp: str
    order_status: OrderStatus
    refund_status: RefundStatus
    # The order total — exact integer cents; shown as dollars.
    amount: Money

    # What was bought.
    product_name: str = ""
    product_sku: str = ""
    quantity: int = 1
    unit_price: Money = Field(default_factory=Money.zero)
    currency: str = "USD"

    # Where it is. All optional, because each only exists at some point in
    # an order's life: a pending order has no carrier, a cancelled one has
    # no ETA, and a delivered one has a delivery date rather than a
    # location in transit.
    carrier: str | None = None
    tracking_number: str | None = None
    estimated_delivery_date: str | None = None
    delivered_at: str | None = None
    current_location: str | None = None

    @property
    def is_trackable(self) -> bool:
        """Whether there is anything to say about where this order is."""
        return bool(self.tracking_number or self.current_location or self.delivered_at)

    @property
    def whereabouts(self) -> str:
        """Where this order has got to, in words a customer can read.

        Deterministic, and deliberately so. "Where is my order" is
        answerable from the record; routing it through a model would be
        inviting an invented delivery date.
        """
        if self.order_status == "delivered" and self.delivered_at:
            return f"delivered on {self.delivered_at[:10]}"
        if self.order_status == "cancelled":
            return "cancelled, so it is not on its way"
        if self.current_location and self.estimated_delivery_date:
            return (
                f"in transit, last seen at {self.current_location}, "
                f"due {self.estimated_delivery_date[:10]}"
            )
        if self.current_location:
            return f"in transit, last seen at {self.current_location}"
        if self.estimated_delivery_date:
            return f"not yet shipped, due {self.estimated_delivery_date[:10]}"
        return "not yet shipped"

    @property
    def description(self) -> str:
        """What this order is, for a line a person reads."""
        if not self.product_name:
            return f"Order {self.order_id}"
        quantity = f"{self.quantity} x " if self.quantity > 1 else ""
        return f"{quantity}{self.product_name}"

    @property
    def already_refunded(self) -> bool:
        """Whether this order has been refunded and cannot be refunded again."""
        return self.refund_status == "completed"

    @property
    def is_refund_eligible(self) -> bool:
        """Whether this order has reached a state a refund can apply to.

        An order becomes refundable once it is settled: delivered, so the
        customer has it and wants their money back, or cancelled before
        delivery, so they never received it. A pending or shipped order
        is still in flight and there is nothing to refund yet.
        """
        return self.order_status in REFUNDABLE_ORDER_STATUSES

    @property
    def refund_ineligibility_reason(self) -> str:
        """Why this order cannot be refunded, in words a customer can read."""
        if self.already_refunded:
            return f"Order {self.order_id} has already been refunded in full"
        if not self.is_refund_eligible:
            return (
                f"Order {self.order_id} is still {self.order_status}, so there is "
                f"nothing to refund yet — a refund applies once an order has been "
                f"delivered, or if it is cancelled before delivery"
            )
        return ""

    @property
    def is_cancellable(self) -> bool:
        """Whether this order has not yet left, and so can still be stopped."""
        return self.order_status in CANCELLABLE_ORDER_STATUSES

    @property
    def cancellation_ineligibility_reason(self) -> str:
        """Why this order cannot be cancelled, in words a customer can read."""
        if self.is_cancellable:
            return ""
        return (
            f"Order {self.order_id} is already {self.order_status}, so it can no longer "
            "be cancelled — only an order that has not shipped yet can be"
        )


class User(BaseModel):
    """One customer.

    `user_code` is the credential the customer types; `user_id` is the
    internal key that appears in joins and logs. Keeping them separate
    means an enumerable primary key is never the thing that grants
    access.
    """

    user_id: int
    user_code: str
    created_at: str
    name: str
    email: str
    phone: str | None = None

    @property
    def masked_email(self) -> str:
        """Email with the local part partly hidden, for read-back."""
        local, _, domain = self.email.partition("@")
        if not domain:
            return self.email
        shown = local[:2] if len(local) > 2 else local[:1]
        return f"{shown}{'*' * max(3, len(local) - len(shown))}@{domain}"


class HitlRequest(BaseModel):
    """A refund waiting on a human decision."""

    id: int | None = None
    order_id: int
    user_id: int
    requested_amount: Money
    action: ReviewAction = "refund"
    status: HitlStatus = "pending"
    created_at: str = Field(default_factory=lambda: utc_now().isoformat())
