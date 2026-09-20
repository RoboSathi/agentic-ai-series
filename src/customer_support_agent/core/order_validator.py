"""Customer Support's implementation of the generic BasicSanity contract.

This module deliberately owns Customer Support domain knowledge. It knows
about orders, customers and refund state; ``agentic_commons`` does not.

The checker runs before the LLM and only performs deterministic work. When an
explicit order id is present, it looks up the customer's own order and can
reject requests that are already known to be impossible, such as refunding an
order that has already been fully refunded.
"""

import re
from collections.abc import Iterable

from agentic_commons.basic_sanity import (
    BasicSanityChecker,
    BasicSanityFinding,
    BasicSanityRequest,
    BasicSanityResult,
)
from agentic_commons.logging import get_logger
from customer_support_agent.config import get_agent_settings
from customer_support_agent.core.repositories import OrdersRepository


# Deliberately conservative: only treat a number as an order id when the
# customer explicitly labels it as an order. Bare numbers are left to the
# existing menu-selection path so BasicSanity never guesses.
_ORDER_ID_PATTERNS = (
    # Explicit order references: "order 10891", "order #10891",
    # "order id 10891", "order number: 10891".
    re.compile(
        r"\border(?:\s+(?:id|number))?\s*[#:]?\s*(?P<order_id>\d+)\b",
        re.IGNORECASE,
    ),
    # Refund requests in the compact form customers naturally use: "refund 10891"
    # or "refund order 10891". Treat long numeric tokens as order ids here;
    # currency/amount wording is deliberately excluded so "refund $50" and
    # "refund 50 dollars" still require the model to resolve the request.
    re.compile(
        r"\brefund\b(?:\s+(?:for|on))?(?:\s+order)?\s*[#:]?\s*"
        r"(?P<order_id>\d{4,})\b(?!\s*(?:dollars?|usd|\$))",
        re.IGNORECASE,
    ),
)


class OrderSanityCheck(BasicSanityChecker):
    """Deterministic Customer Support preflight for explicit order requests."""

    def __init__(
        self,
        orders: OrdersRepository,
        *,
        refund_markers: Iterable[str] | None = None,
    ) -> None:
        self._orders = orders
        self._refund_markers = tuple(
            marker.casefold()
            for marker in (
                refund_markers
                if refund_markers is not None
                else _configured_refund_markers()
            )
            if marker.strip()
        )
        self._logger = get_logger("customer_support_agent")

    def check(self, request: BasicSanityRequest) -> BasicSanityResult:
        """Run only deterministic checks that are safe before the model."""
        order_id = self._extract_order_id(request.query)
        if order_id is None:
            return BasicSanityResult(
                passed=True,
                reason_code="NO_EXPLICIT_ORDER_ID",
                findings=[
                    BasicSanityFinding(
                        check="explicit_order_id",
                        passed=True,
                        reason="No explicit order id was present; no order lookup was attempted.",
                    )
                ],
                facts={"order_id": None, "operation": self._operation(request.query)},
            )

        if request.customer_id is None:
            # Fail closed. The normal graph authenticates before this node,
            # but the component remains safe when invoked independently.
            return BasicSanityResult(
                passed=False,
                reason_code="CUSTOMER_NOT_AUTHENTICATED",
                message="I need to verify your account before I can look up that order.",
                findings=[
                    BasicSanityFinding(
                        check="customer_authenticated",
                        passed=False,
                        reason="No authenticated customer id was supplied.",
                        evidence={"order_id": order_id},
                    )
                ],
                facts={"order_id": order_id},
            )

        order = self._orders.get_for_customer(order_id, request.customer_id)
        if order is None:
            return BasicSanityResult(
                passed=False,
                reason_code="ORDER_NOT_FOUND_FOR_CUSTOMER",
                message=f"I couldn't find order {order_id} on your account.",
                findings=[
                    BasicSanityFinding(
                        check="order_exists_and_is_owned",
                        passed=False,
                        reason="The order does not exist for the authenticated customer.",
                        evidence={
                            "order_id": order_id,
                            "customer_id": request.customer_id,
                        },
                    )
                ],
                facts={"order_id": order_id, "operation": self._operation(request.query)},
            )

        operation = self._operation(request.query)
        facts = {
            "order_id": order.order_id,
            "customer_id": order.user_id,
            "order_status": order.order_status,
            "refund_status": order.refund_status,
            "amount": order.amount,
            "already_refunded": order.already_refunded,
            "refund_eligible": order.is_refund_eligible,
            "operation": operation,
        }
        findings = [
            BasicSanityFinding(
                check="order_exists_and_is_owned",
                passed=True,
                reason="Order exists and belongs to the authenticated customer.",
                evidence={
                    "order_id": order.order_id,
                    "customer_id": request.customer_id,
                },
            )
        ]

        if operation == "refund_request" and order.already_refunded:
            self._logger.info(
                "BasicSanity blocked an already-refunded order",
                order_id=order.order_id,
                customer_id=request.customer_id,
            )
            findings.append(
                BasicSanityFinding(
                    check="not_already_refunded",
                    passed=False,
                    reason="The order has already been refunded in full.",
                    evidence={
                        "order_id": order.order_id,
                        "refund_status": order.refund_status,
                    },
                )
            )
            return BasicSanityResult(
                passed=False,
                reason_code="ORDER_ALREADY_REFUNDED",
                message=f"Order {order.order_id} has already been refunded in full.",
                findings=findings,
                facts=facts,
            )

        if operation == "refund_request" and not order.is_refund_eligible:
            findings.append(
                BasicSanityFinding(
                    check="refund_eligible",
                    passed=False,
                    reason=order.refund_ineligibility_reason,
                    evidence={
                        "order_id": order.order_id,
                        "order_status": order.order_status,
                        "refund_status": order.refund_status,
                    },
                )
            )
            return BasicSanityResult(
                passed=False,
                reason_code="ORDER_NOT_REFUND_ELIGIBLE",
                message=order.refund_ineligibility_reason,
                findings=findings,
                facts=facts,
            )

        if operation == "refund_request":
            findings.append(
                BasicSanityFinding(
                    check="not_already_refunded",
                    passed=True,
                    reason="The order is not fully refunded.",
                    evidence={"refund_status": order.refund_status},
                )
            )
            findings.append(
                BasicSanityFinding(
                    check="refund_eligible",
                    passed=True,
                    reason="The order is in a refundable state.",
                    evidence={"order_status": order.order_status},
                )
            )

        return BasicSanityResult(
            passed=True,
            reason_code="BASIC_SANITY_PASSED",
            findings=findings,
            facts=facts,
        )

    @staticmethod
    def _extract_order_id(query: str) -> int | None:
        """Extract an order id from an explicit order/refund reference.

        The normal form is an order-labelled number. Customer Support also
        accepts the compact refund phrasing "refund 10891" because that is
        an explicit refund request for a long numeric token in this domain.
        Amount-looking forms such as "refund $50" remain unresolved here.
        """
        for pattern in _ORDER_ID_PATTERNS:
            match = pattern.search(query)
            if match is not None:
                return int(match.group("order_id"))
        return None

    def _operation(self, query: str) -> str:
        """Detect only the narrow operation needed for refund preflight."""
        lowered = query.casefold()
        return (
            "refund_request"
            if any(marker in lowered for marker in self._refund_markers)
            else "unknown"
        )


def _configured_refund_markers() -> list[str]:
    """Read refund phrases from the Customer Support agent's own config."""
    for intent in get_agent_settings().intents:
        if intent.name == "refund_request":
            return list(intent.phrases)
    return []
