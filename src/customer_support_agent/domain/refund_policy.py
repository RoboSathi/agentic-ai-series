"""Single source of truth for customer-support refund business rules.

The Gatekeeper, BasicSanity, tool handlers and the refund Saga need the same
facts to agree on what a refund means. This module owns those domain rules and
returns a domain-level result; each caller remains responsible for presenting
that result through its own contract (sanity findings, governance checks, or
tool errors).
"""

import math
from typing import Literal

from pydantic import BaseModel

from customer_support_agent.domain.models import Order
from customer_support_agent.persistence.repositories import OrdersRepository

RefundAction = Literal["request", "process_refund", "queue_for_approval"]


class RefundValidationResult(BaseModel):
    """The outcome of the authoritative refund-policy evaluation."""

    allowed: bool
    reason_code: str = "REFUND_POLICY_PASSED"
    reason: str = "all applicable refund business rules passed"
    order: Order | None = None
    amount: float | None = None


class RefundPolicy:
    """Authoritative, reusable refund business rules for this agent."""

    def __init__(self, orders: OrdersRepository) -> None:
        self._orders = orders

    def validate(
        self,
        order_id: int,
        customer_id: int,
        amount: float | None = None,
        *,
        action: RefundAction = "request",
    ) -> RefundValidationResult:
        """Validate ownership, state and amount without making any side effect."""
        order = self._orders.get_for_customer(order_id, customer_id)
        if order is None:
            return self._reject(
                "ORDER_NOT_FOUND_FOR_CUSTOMER",
                f"Order {order_id} was not found on the authenticated customer's account.",
            )

        if order.already_refunded:
            return self._reject(
                "ORDER_ALREADY_REFUNDED",
                f"Order {order_id} has already been refunded in full.",
                order=order,
            )

        if not order.is_refund_eligible:
            return self._reject(
                "ORDER_NOT_REFUND_ELIGIBLE",
                order.refund_ineligibility_reason,
                order=order,
            )

        if amount is None:
            return RefundValidationResult(allowed=True, order=order)

        if not isinstance(amount, (int, float)) or isinstance(amount, bool) or not math.isfinite(float(amount)):
            return self._reject(
                "REFUND_AMOUNT_INVALID",
                "The refund amount must be a finite numeric value.",
                order=order,
            )

        normalized = float(amount)
        if normalized <= 0:
            return self._reject(
                "INVALID_REFUND_AMOUNT",
                "The refund amount must be greater than zero.",
                order=order,
                amount=normalized,
            )

        if normalized > float(order.amount):
            return self._reject(
                "REFUND_AMOUNT_EXCEEDS_ORDER_TOTAL",
                f"Requested refund ${normalized:,.2f} exceeds order {order_id}'s ${order.amount:,.2f} total.",
                order=order,
                amount=normalized,
            )

        if action == "process_refund" and not math.isclose(
            normalized, float(order.amount), rel_tol=0.0, abs_tol=0.01
        ):
            return self._reject(
                "PARTIAL_REFUND_NOT_SUPPORTED",
                f"Direct refund for order {order_id} must equal the authoritative order total of ${order.amount:,.2f}.",
                order=order,
                amount=normalized,
            )

        return RefundValidationResult(
            allowed=True,
            order=order,
            amount=normalized,
        )

    @staticmethod
    def _reject(
        reason_code: str,
        reason: str,
        *,
        order: Order | None = None,
        amount: float | None = None,
    ) -> RefundValidationResult:
        return RefundValidationResult(
            allowed=False,
            reason_code=reason_code,
            reason=reason,
            order=order,
            amount=amount,
        )
