"""What this business is made of (LLD §21).

The domain vocabulary is deliberately narrow: an Order, a User, and a
request waiting on a human. Everything else — authority, memory,
context, resilience — belongs to `agentic_commons` and arrives by
injection.

Nothing here reaches a database or a model. These are the nouns; the
verbs live in `tools/`, the tables in `persistence/`, and the state of a
turn in `graph/state.py`.
"""


from typing import Literal

from pydantic import BaseModel, Field

from agentic_commons.observability.types import utc_now

OrderStatus = Literal["pending", "shipped", "delivered", "cancelled"]
RefundStatus = Literal["none", "pending_review", "approved", "denied", "completed"]
HitlStatus = Literal["pending", "approved", "denied"]

# An order is refundable only once it is settled. Domain invariant, not a
# tunable: it is what "refund" means for this business, and the dummy
# data generator applies the same rule so the fixtures cannot contradict
# the policy the agent is asked to enforce.
REFUNDABLE_ORDER_STATUSES: frozenset[str] = frozenset({"delivered", "cancelled"})

class Order(BaseModel):
    """One customer order."""

    order_id: int
    user_id: int
    timestamp: str
    order_status: OrderStatus
    refund_status: RefundStatus
    amount: float

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
    requested_amount: float
    status: HitlStatus = "pending"
    created_at: str = Field(default_factory=lambda: utc_now().isoformat())
