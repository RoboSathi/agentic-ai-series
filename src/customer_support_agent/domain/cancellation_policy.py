"""Business rules for cancelling an order that has not shipped.

A cancellation here is also a refund: the customer paid for an order that
will now never arrive, so the full total goes back in the same step. That
makes it a money-moving action, bound by the same authority rules as a
refund — but those live in the Gatekeeper. This module answers only what
makes a cancellation *correct*, and those rules hold with the harness off.
"""

from typing import Any

from agent_platform.core.primitives.money import InvalidAmountError, Money
from customer_support_agent.domain.models import Order
from customer_support_agent.domain.ports import OrderLookup
from customer_support_agent.domain.refund_policy import RefundValidationResult

_CANCELLED = "cancelled"
# Asked to cancel an order that already is, the customer is really asking
# where things stand — so the answer states the refund too, from the record.
# Said here, deterministically, it leaves nothing for a model to paraphrase.
_ALREADY_CANCELLED = "Order {order_id} is already cancelled"
_REFUND_STATE_AFTER_CANCEL: dict[str, str] = {
    "completed": ", and its {amount} refund has been completed.",
    "issued": ", and its {amount} refund has been issued and is waiting to settle.",
    "pending_review": ", and its {amount} refund is awaiting review.",
    "denied": "; its {amount} refund was declined by a reviewer.",
    "none": "; no refund has been issued for it yet.",
}


class CancellationPolicy:
    """Ownership, state and amount rules for cancel-and-refund."""

    def __init__(self, orders: OrderLookup) -> None:
        self._orders = orders

    def validate(
        self, order_id: int, customer_id: int, amount: Any = None
    ) -> RefundValidationResult:
        """Validate without side effects; `amount` is checked only when given."""
        order = self._orders.get_for_customer(order_id, customer_id)
        if order is None:
            return _reject(
                "ORDER_NOT_FOUND_FOR_CUSTOMER",
                f"Order {order_id} was not found on the authenticated customer's account.",
            )
        if order.order_status == _CANCELLED:
            return _reject("ORDER_ALREADY_CANCELLED", _already_cancelled(order), order=order)
        if order.already_refunded:
            return _reject(
                "ORDER_ALREADY_REFUNDED",
                f"Order {order_id} has already been refunded in full.",
                order=order,
            )
        if not order.is_cancellable:
            return _reject(
                "ORDER_NOT_CANCELLABLE", order.cancellation_ineligibility_reason, order=order
            )
        if amount is None:
            return RefundValidationResult(allowed=True, order=order)
        return _validate_amount(order, amount)


def _validate_amount(order: Order, amount: Any) -> RefundValidationResult:
    """A cancellation refunds the full order total, exactly to the cent."""
    try:
        requested = Money.parse(amount)
    except InvalidAmountError as exc:
        return _reject(
            "REFUND_AMOUNT_INVALID", f"The refund amount is invalid: {exc}.", order=order
        )
    if requested != order.amount:
        return _reject(
            "PARTIAL_REFUND_NOT_SUPPORTED",
            f"Requested {requested} for order {order.order_id}, but cancelling refunds the "
            f"full order total of {order.amount}.",
            order=order,
            amount=requested,
        )
    return RefundValidationResult(allowed=True, order=order, amount=requested)


def _already_cancelled(order: Order) -> str:
    """Where a cancelled order stands: cancelled, and what became of its refund."""
    head = _ALREADY_CANCELLED.format(order_id=order.order_id)
    tail = _REFUND_STATE_AFTER_CANCEL.get(order.refund_status, ".")
    return head + tail.format(amount=order.amount)


def _reject(
    reason_code: str,
    reason: str,
    *,
    order: Order | None = None,
    amount: Money | None = None,
) -> RefundValidationResult:
    return RefundValidationResult(
        allowed=False, reason_code=reason_code, reason=reason, order=order, amount=amount
    )
