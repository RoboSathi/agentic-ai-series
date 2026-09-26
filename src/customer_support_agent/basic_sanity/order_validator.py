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
from agentic_commons.money import Money, money_or_none
from customer_support_agent.config import get_agent_settings
from customer_support_agent.domain.order_reference import ORDER_UNDER_DISCUSSION
from customer_support_agent.domain.refund_policy import RefundPolicy
from customer_support_agent.persistence.repositories import OrdersRepository

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
        refund_policy: RefundPolicy | None = None,
        refund_markers: Iterable[str] | None = None,
    ) -> None:
        self._orders = orders
        self._refund_policy = refund_policy or RefundPolicy(orders)
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
        order_id = self.extract_order_id(request.query, request.metadata)
        operation = self._operation(request.query, request.metadata)
        # The platform hands over an opaque subject id; reading it back as
        # a customer number is this agent's business, not the platform's.
        customer_id = int(request.subject_id) if request.subject_id.isdigit() else None
        if order_id is None:
            return BasicSanityResult(
                passed=True,
                reason_code="ORDER_REFERENCE_NOT_PRESENT",
                message=(
                    "No deterministically recognizable order reference was found; "
                    "order-state validation is deferred until an order is identified."
                ),
                findings=[
                    BasicSanityFinding(
                        check="order_reference",
                        passed=True,
                        applicable=False,
                        reason=(
                            "No explicit order reference was supplied. BasicSanity does "
                            "not guess or autocorrect identifiers."
                        ),
                        evidence={
                            "operation": operation,
                            "order_id_present": False,
                            "lookup_attempted": False,
                        },
                    )
                ],
                facts={
                    "order_id": None,
                    "operation": operation,
                    "order_id_present": False,
                    "lookup_attempted": False,
                },
            )

        if customer_id is None:
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

        if operation == "REQUEST_REFUND":
            requested_amount = _requested_amount(request.metadata)
            validation = self._refund_policy.validate(
                order_id=order_id,
                customer_id=customer_id,
                amount=requested_amount,
            )
            order = validation.order
            if order is None:
                return BasicSanityResult(
                    passed=False,
                    reason_code=validation.reason_code,
                    message=validation.reason,
                    findings=[
                        BasicSanityFinding(
                            check="refund_policy",
                            passed=False,
                            reason=validation.reason,
                            evidence={
                                "order_id": order_id,
                                "customer_id": customer_id,
                            },
                        )
                    ],
                    facts={
                        "order_id": order_id,
                        "operation": operation,
                        **({"requested_amount": str(requested_amount)} if requested_amount is not None else {}),
                    },
                )
        else:
            order = self._orders.get_for_customer(order_id, customer_id)
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
                                "customer_id": customer_id,
                            },
                        )
                    ],
                    facts={"order_id": order_id, "operation": operation},
                )

        facts = {
            "order_id": order.order_id,
            "customer_id": order.user_id,
            "order_status": order.order_status,
            "refund_status": order.refund_status,
            "amount": str(order.amount),
            # Established here so a "where is it" question can be answered
            # from the record rather than left for the model to guess at.
            "product_name": order.product_name,
            "whereabouts": order.whereabouts,
            "estimated_delivery_date": order.estimated_delivery_date,
            "tracking_number": order.tracking_number,
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
                    "customer_id": customer_id,
                    "order_exists": True,
                    "customer_owns_order": True,
                },
            )
        ]

        if operation == "REQUEST_REFUND":
            requested_amount = _requested_amount(request.metadata)
            if requested_amount is not None:
                facts["requested_amount"] = str(requested_amount)

            if not validation.allowed:
                finding_map = {
                    "ORDER_ALREADY_REFUNDED": (
                        "not_already_refunded",
                        {"order_id": order_id, "refund_status": order.refund_status},
                    ),
                    "ORDER_NOT_REFUND_ELIGIBLE": (
                        "refund_eligible",
                        {
                            "order_id": order_id,
                            "order_status": order.order_status,
                            "refund_status": order.refund_status,
                        },
                    ),
                    "INVALID_REFUND_AMOUNT": (
                        "requested_amount_positive",
                        {"requested_amount": str(requested_amount)},
                    ),
                    "REFUND_AMOUNT_EXCEEDS_ORDER_TOTAL": (
                        "requested_amount_within_order_total",
                        {
                            "requested_amount": str(requested_amount),
                            "order_amount": str(order.amount),
                        },
                    ),
                    "PARTIAL_REFUND_NOT_SUPPORTED": (
                        "full_refund_only",
                        {
                            "requested_amount": str(requested_amount),
                            "order_amount": str(order.amount),
                        },
                    ),
                }
                check_name, evidence = finding_map.get(
                    validation.reason_code,
                    ("refund_policy", {"order_id": order_id}),
                )
                findings.append(
                    BasicSanityFinding(
                        check=check_name,
                        passed=False,
                        reason=validation.reason,
                        evidence=evidence,
                    )
                )
                return BasicSanityResult(
                    passed=False,
                    reason_code=validation.reason_code,
                    message=validation.reason,
                    findings=findings,
                    facts=facts,
                )

            if requested_amount is not None:
                findings.append(
                    BasicSanityFinding(
                        check="requested_amount_positive",
                        passed=True,
                        reason="The requested refund amount is greater than zero.",
                        evidence={"requested_amount": str(requested_amount)},
                    )
                )
                findings.append(
                    BasicSanityFinding(
                        check="requested_amount_within_order_total",
                        passed=True,
                        reason="The requested refund does not exceed the order total.",
                        evidence={
                            "requested_amount": str(requested_amount),
                            "order_amount": str(order.amount),
                            "within_order_total": True,
                        },
                    )
                )

            findings.extend(
                [
                    BasicSanityFinding(
                        check="not_already_refunded",
                        passed=True,
                        reason="The order is not fully refunded.",
                        evidence={
                            "refund_status": order.refund_status,
                            "already_refunded": False,
                        },
                    ),
                    BasicSanityFinding(
                        check="refund_eligible",
                        passed=True,
                        reason="The order is in a refundable state.",
                        evidence={
                            "order_status": order.order_status,
                            "refund_eligible": True,
                        },
                    ),
                ]
            )

        return BasicSanityResult(
            passed=True,
            reason_code="BASIC_SANITY_PASSED",
            findings=findings,
            facts=facts,
        )

    @staticmethod
    def extract_order_id(query: str, metadata: dict[str, object] | None = None) -> int | None:
        """Extract an order id from the user query, the turn, or an exact call.

        Still no guessing. Both fallbacks are references somebody else
        already established: ``order_under_discussion`` is the order this
        turn resolved from what the customer said, and the tool-call
        fallback — used only by the Gatekeeper's final re-check — is an
        ``order_id`` the agent has already validated. Neither is a number
        inferred from arbitrary text.

        Without the first of these, "I want a refund" one turn after
        "what about order 10301?" reads as a refund request naming no
        order, and the deterministic eligibility checks are skipped on
        exactly the turn that moves money.
        """
        for pattern in _ORDER_ID_PATTERNS:
            match = pattern.search(query)
            if match is not None:
                return int(match.group("order_id"))
        carried = (metadata or {}).get(ORDER_UNDER_DISCUSSION)
        if isinstance(carried, int) and not isinstance(carried, bool) and carried > 0:
            return carried
        tool_params = (metadata or {}).get("tool_params")
        if isinstance(tool_params, dict):
            candidate = tool_params.get("order_id")
            if isinstance(candidate, int) and candidate > 0:
                return candidate
            if isinstance(candidate, str) and candidate.isdigit():
                return int(candidate)
        return None

    def _operation(self, query: str, metadata: dict[str, object] | None = None) -> str:
        """Detect the refund operation from user wording or exact tool identity."""
        tool_action = (metadata or {}).get("tool_action", "")
        bare_action = str(tool_action).rsplit(".", maxsplit=1)[-1]
        if bare_action in {"process_refund", "queue_for_approval"}:
            return "REQUEST_REFUND"
        lowered = query.casefold()
        return (
            "REQUEST_REFUND"
            if any(marker in lowered for marker in self._refund_markers)
            else "unknown"
        )


def _requested_amount(metadata: dict[str, object] | None) -> Money | None:
    """Read an already-validated tool amount, exactly, when the Gatekeeper rechecks a call."""
    params = (metadata or {}).get("tool_params")
    if not isinstance(params, dict):
        return None
    return money_or_none(params.get("amount"))


def _configured_refund_markers() -> list[str]:
    """Read refund phrases from the Customer Support agent's own config."""
    for intent in get_agent_settings().intents:
        if intent.name == "REQUEST_REFUND":
            return list(intent.phrases)
    return []
