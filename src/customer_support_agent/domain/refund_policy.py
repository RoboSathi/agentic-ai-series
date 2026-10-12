"""Single source of truth for customer-support refund business rules.

The Gatekeeper, DeterministicCheck, tool handlers and the refund Saga need the same
facts to agree on what a refund means. This module owns those domain rules and
returns a domain-level result; each caller remains responsible for presenting
that result through its own contract (deterministic-check findings, governance checks, or
tool errors).
"""

from typing import Any, Literal

from pydantic import BaseModel

from agent_platform.core.primitives.money import InvalidAmountError, Money
from customer_support_agent.domain.models import Order
from customer_support_agent.domain.ports import OrderLookup

RefundAction = Literal["request", "process_refund", "queue_for_approval"]


# A refund a reviewer declined cannot be requested again.
REFUND_DECLINED_FINAL = "REFUND_DECLINED_FINAL"


class RefundValidationResult(BaseModel):
    """The outcome of the authoritative refund-policy evaluation."""

    allowed: bool
    reason_code: str = "REFUND_POLICY_PASSED"
    reason: str = "all applicable refund business rules passed"
    order: Order | None = None
    amount: Money | None = None


class RefundPolicy:
    """Authoritative, reusable refund business rules for this agent."""

    def __init__(self, orders: OrderLookup, direct_limit: Money | None = None) -> None:
        self._orders = orders
        # The most this agent may refund without a person — from its
        # persisted AgentPolicy. None: not known here — the queue is not
        # blocked on amount alone (review is never the unsafe direction).
        self._direct_limit = direct_limit

    def validate(
        self,
        order_id: int,
        customer_id: int,
        amount: Any = None,
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

        if order.refund_declined:
            return self._reject(
                REFUND_DECLINED_FINAL, order.refund_ineligibility_reason, order=order
            )

        if not order.is_refund_eligible:
            return self._reject(
                "ORDER_NOT_REFUND_ELIGIBLE",
                order.refund_ineligibility_reason,
                order=order,
            )

        if amount is None:
            return RefundValidationResult(allowed=True, order=order)
        try:
            requested = Money.parse(amount)
        except InvalidAmountError as exc:
            return self._reject(
                "REFUND_AMOUNT_INVALID", f"The refund amount is invalid: {exc}.", order=order
            )
        return self._validate_amount(order, requested, action)

    def _validate_amount(
        self, order: Order, requested: Money, action: RefundAction
    ) -> RefundValidationResult:
        """Amount rules, exact to the cent: full refunds only, and review only above the limit."""
        if not requested:
            return self._reject(
                "INVALID_REFUND_AMOUNT",
                "The refund amount must be greater than zero.",
                order=order,
                amount=requested,
            )
        if requested > order.amount:
            return self._reject(
                "REFUND_AMOUNT_EXCEEDS_ORDER_TOTAL",
                f"Requested refund {requested} exceeds order {order.order_id}'s {order.amount} total.",
                order=order,
                amount=requested,
            )
        if action in ("process_refund", "queue_for_approval") and requested != order.amount:
            return self._reject(
                "PARTIAL_REFUND_NOT_SUPPORTED",
                f"Requested {requested} for order {order.order_id}, but a refund must be for the "
                f"full order total of {order.amount}; partial refunds are not supported.",
                order=order,
                amount=requested,
            )
        if action == "queue_for_approval" and not self.needs_review(requested):
            return self._reject(
                "WITHIN_DIRECT_LIMIT",
                f"A {requested} refund is within the {self._direct_limit} direct-refund limit, so it "
                "needs no human review; process it directly.",
                order=order,
                amount=requested,
            )
        return RefundValidationResult(allowed=True, order=order, amount=requested)

    def needs_review(self, requested: Money) -> bool:
        """Whether an amount is above what this agent may refund on its own.

        With no declared limit the question cannot be decided here, and
        sending a refund to a person is never the unsafe direction — so an
        unknown limit does not block the queue; only a known one does.
        """
        return self._direct_limit is None or requested > self._direct_limit

    @staticmethod
    def _reject(
        reason_code: str,
        reason: str,
        *,
        order: Order | None = None,
        amount: Money | None = None,
    ) -> RefundValidationResult:
        return RefundValidationResult(
            allowed=False,
            reason_code=reason_code,
            reason=reason,
            order=order,
            amount=amount,
        )
