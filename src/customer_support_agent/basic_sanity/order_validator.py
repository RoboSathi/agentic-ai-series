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
from typing import Any

from agentic_commons.basic_sanity import (
    BasicSanityChecker,
    BasicSanityFinding,
    BasicSanityRequest,
    BasicSanityResult,
)
from agentic_commons.logging import get_logger
from agentic_commons.money import Money, money_or_none
from customer_support_agent.config import get_agent_settings
from customer_support_agent.domain.cancellation_policy import CancellationPolicy
from customer_support_agent.domain.models import Order
from customer_support_agent.domain.order_reference import (
    OrderReferenceResolver,
)
from customer_support_agent.domain.refund_policy import RefundPolicy
from customer_support_agent.persistence.repositories import OrdersRepository

# The finding that proves an order exists and belongs to this customer. A
# request about that order may still be refused, but the order it names is
# settled, and the conversation is now about it.
ORDER_OWNED_CHECK = "order_exists_and_is_owned"
_REFUND_OPERATION = "REQUEST_REFUND"
# Asking where a refund stands is a read: ownership is checked, eligibility
# is not — "when will I get my refund" must not be refused as a second request.
REFUND_STATUS_OPERATION = "REFUND_STATUS"
# The cancel operation, named as the intent router names it, and the tool
# that performs it.
CANCEL_OPERATION = "CANCEL_ORDER"
_CANCEL_TOOL = "cancel_order"
_CANCELLABLE_CHECK = "order_cancellable"
# "cancel 10022": the compact form no configured phrase covers.
_CANCEL_BY_ID = re.compile(r"\bcancel\s+(?:order\s*)?#?\s*\d{4,}\b", re.IGNORECASE)


class OrderSanityCheck(BasicSanityChecker):
    """Deterministic Customer Support preflight for explicit order requests."""

    def __init__(
        self,
        orders: OrdersRepository,
        *,
        refund_policy: RefundPolicy | None = None,
        cancellation_policy: CancellationPolicy | None = None,
        refund_markers: Iterable[str] | None = None,
        references: OrderReferenceResolver | None = None,
    ) -> None:
        self._orders = orders
        # The one reader of which order a message names. BasicSanity used to
        # carry its own, and two readers that disagree are how an agent
        # validates one order and acts on another.
        self._references = references or OrderReferenceResolver.from_orders(orders)
        self._refund_policy = refund_policy or RefundPolicy(orders)
        self._cancellation_policy = cancellation_policy or CancellationPolicy(orders)
        self._cancel_markers = _markers(_configured_phrases(CANCEL_OPERATION))
        self._status_markers = _markers(_configured_phrases(REFUND_STATUS_OPERATION))
        self._refund_markers = _markers(
            refund_markers if refund_markers is not None else _configured_phrases(_REFUND_OPERATION)
        )
        self._logger = get_logger("customer_support_agent")

    def check(self, request: BasicSanityRequest) -> BasicSanityResult:
        """Run only deterministic checks that are safe before the model."""
        # The platform hands over an opaque subject id; reading it back as
        # a customer number is this agent's business, not the platform's.
        customer_id = int(request.subject_id) if request.subject_id.isdigit() else None
        order_id = self.order_id(request.query, request.metadata, customer_id)
        operation = self._operation(request.query, request.metadata)
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
                        **(
                            {"requested_amount": str(requested_amount)}
                            if requested_amount is not None
                            else {}
                        ),
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
                            check=ORDER_OWNED_CHECK,
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
                check=ORDER_OWNED_CHECK,
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

        if operation == CANCEL_OPERATION:
            return self._check_cancellation(order, customer_id, request, findings, facts)

        if operation == "REQUEST_REFUND":
            requested_amount = _requested_amount(request.metadata)
            if requested_amount is not None:
                facts["requested_amount"] = str(requested_amount)

            if not validation.allowed:
                finding_map: dict[str, tuple[str, dict[str, Any]]] = {
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

    def order_id(
        self, query: str, metadata: dict[str, object] | None, customer_id: int | None
    ) -> int | None:
        """The order this request names, the turn settled, or an exact call re-checked.

        Still no guessing. The message is read by the shared resolver; the
        fallback — used only by the Gatekeeper's final re-check — is an
        ``order_id`` the agent has already validated, not a number inferred
        from text.

        Without the carried reference, "I want a refund" one turn after
        "what about order 10301?" reads as a refund request naming no
        order, and the deterministic eligibility checks are skipped on
        exactly the turn that moves money.
        """
        resolved = self._references.resolve(query, metadata, None, customer_id=customer_id)
        if resolved is not None:
            return resolved
        tool_params = (metadata or {}).get("tool_params")
        if not isinstance(tool_params, dict):
            return None
        candidate = tool_params.get("order_id")
        if isinstance(candidate, int) and candidate > 0:
            return candidate
        return int(candidate) if isinstance(candidate, str) and candidate.isdigit() else None

    def _check_cancellation(
        self,
        order: Order,
        customer_id: int,
        request: BasicSanityRequest,
        findings: list[BasicSanityFinding],
        facts: dict[str, Any],
    ) -> BasicSanityResult:
        """Whether this order can be cancelled — the same rules the Gatekeeper and the tool apply.

        An order already cancelled, shipped or delivered is answered here,
        from the record, before a model is asked to describe its state.
        """
        validation = self._cancellation_policy.validate(
            order.order_id, customer_id, _requested_amount(request.metadata)
        )
        findings.append(
            BasicSanityFinding(
                check=_CANCELLABLE_CHECK,
                passed=validation.allowed,
                reason=validation.reason or "The order has not shipped and can be cancelled.",
                evidence={
                    "order_id": order.order_id,
                    "order_status": order.order_status,
                    "refund_status": order.refund_status,
                },
            )
        )
        if not validation.allowed:
            return BasicSanityResult(
                passed=False,
                reason_code=validation.reason_code,
                message=validation.reason,
                findings=findings,
                facts=facts,
            )
        return BasicSanityResult(
            passed=True, reason_code="BASIC_SANITY_PASSED", findings=findings, facts=facts
        )

    def _operation(self, query: str, metadata: dict[str, object] | None = None) -> str:
        """Detect the operation from user wording or exact tool identity.

        Cancellation is tested first: "cancel my order and refund me" is a
        cancellation, which refunds by itself, not a refund of a pending order.
        """
        tool_action = (metadata or {}).get("tool_action", "")
        bare_action = str(tool_action).rsplit(".", maxsplit=1)[-1]
        if bare_action == _CANCEL_TOOL:
            return CANCEL_OPERATION
        if bare_action in {"process_refund", "queue_for_approval"}:
            return "REQUEST_REFUND"
        lowered = query.casefold()
        if any(marker in lowered for marker in self._cancel_markers) or _CANCEL_BY_ID.search(query):
            return CANCEL_OPERATION
        if any(marker in lowered for marker in self._status_markers):
            return REFUND_STATUS_OPERATION
        return (
            "REQUEST_REFUND"
            if any(marker in lowered for marker in self._refund_markers)
            else "unknown"
        )


def owned_order_id(result: BasicSanityResult) -> int | None:
    """The order this result proved is the customer's own, whether or not it passed.

    None when nothing was proved: no reference, no customer, or an order
    that is not theirs — which must never become the order under discussion.
    """
    owned = any(f.check == ORDER_OWNED_CHECK and f.passed for f in result.findings)
    order_id = result.facts.get("order_id") if isinstance(result.facts, dict) else None
    return order_id if owned and isinstance(order_id, int) else None


def _requested_amount(metadata: dict[str, object] | None) -> Money | None:
    """Read an already-validated tool amount, exactly, when the Gatekeeper rechecks a call."""
    params = (metadata or {}).get("tool_params")
    if not isinstance(params, dict):
        return None
    return money_or_none(params.get("amount"))


def _configured_phrases(intent_name: str) -> list[str]:
    """An intent's phrases, from the Customer Support agent's own config."""
    for intent in get_agent_settings().intents:
        if intent.name == intent_name:
            return list(intent.phrases)
    return []


def _markers(phrases: Iterable[str]) -> tuple[str, ...]:
    """Phrases as case-folded substrings to look for, blanks dropped."""
    return tuple(phrase.casefold() for phrase in phrases if phrase.strip())
