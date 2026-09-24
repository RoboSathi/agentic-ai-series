"""Customer Support's deterministic business rules for the Gatekeeper."""

import math
from typing import Any

from agentic_commons.gatekeeper.contracts import BusinessRuleResult, DeterministicBusinessRules
from agentic_commons.gatekeeper.types import AuthorityContext, CheckReport
from agentic_commons.security.auth.policy import UNLIMITED, AgentPolicy
from agentic_commons.tools.types import ToolCall
from customer_support_agent.persistence.repositories import HitlQueueRepository, OrdersRepository

# The budget class this agent's money-moving action belongs to. One name,
# used by the seed data, the Gatekeeper ledger and the rule below.
REFUND_ACTION_CLASS = "refund"


class CustomerSupportBusinessRules(DeterministicBusinessRules):
    """Authoritative order/refund checks that must run immediately before writes."""

    def __init__(
        self,
        orders: OrdersRepository,
        hitl: HitlQueueRepository | None = None,
        policy: AgentPolicy | None = None,
    ) -> None:
        self._orders = orders
        self._hitl = hitl
        # This agent's own action class. The platform stores budgets keyed
        # by class and does not know what "refund" means; naming it is this
        # module's job, because this is the module that has refunds.
        self._max_refund_amount = (
            policy.per_action_limit(REFUND_ACTION_CLASS) if policy is not None else UNLIMITED
        )

    def evaluate(self, call: ToolCall, authority: AuthorityContext) -> BusinessRuleResult:
        """Rule on one proposal: who owns the order first, then what may be done to it.

        Ownership is an *authority* question, not a data-access one, so it
        is settled here — before execution — for every action that names an
        order, read or write alike. Leaving reads to a customer-scoped
        repository meant another customer's order was refused by returning
        nothing from a query: the Gatekeeper allowed the call, the executor
        ran it, and the audit trail recorded an authorized read of an order
        the caller had no standing to name. One forgotten `WHERE user_id`
        and that becomes a disclosure. Here it is a denial with a reason.
        """
        action = call.action.rsplit(".", maxsplit=1)[-1]
        is_write = action in {"process_refund", "queue_for_approval"}
        customer_id = authority.request_context.get("customer_id")
        order_id = call.params.get("order_id")
        amount = call.params.get("amount")
        checks: list[CheckReport] = []

        if order_id is None and not is_write:
            return BusinessRuleResult(
                checks=[
                    CheckReport(
                        check_id="gatekeeper.business_rules.not_applicable",
                        name="Agent business rules",
                        status="SKIPPED",
                        reason_code="BUSINESS_RULES_NOT_APPLICABLE",
                        reason=f"'{action}' names no order, so no ownership rule applies.",
                    )
                ]
            )

        if not isinstance(customer_id, int):
            return self._deny(
                checks,
                "CUSTOMER_NOT_AUTHENTICATED",
                "An authenticated customer id is required before an order action can execute.",
                observed={"customer_id_present": False},
            )
        checks.append(CheckReport(
            check_id="gatekeeper.business_rules.customer",
            name="Authenticated customer",
            status="PASS",
            reason_code="CUSTOMER_AUTHENTICATED",
            reason="The action is scoped to an authenticated customer.",
            observed={"customer_id": customer_id},
        ))

        if not isinstance(order_id, int):
            return self._deny(
                checks,
                "ORDER_ID_REQUIRED",
                f"A concrete order id is required for '{action}'.",
                observed={"order_id": order_id},
            )

        order = self._orders.get_for_customer(order_id, customer_id)
        if order is None:
            # Deliberately the same refusal whether the order belongs to
            # somebody else or does not exist. Distinguishing them would
            # turn this gate into an oracle for which order ids are real.
            return self._deny(
                checks,
                "ORDER_NOT_FOUND_FOR_CUSTOMER",
                f"Order {order_id} was not found on the authenticated customer's account.",
                observed={"order_id": order_id, "customer_id": customer_id},
            )
        checks.append(CheckReport(
            check_id="gatekeeper.business_rules.order_ownership",
            name="Order ownership",
            status="PASS",
            reason_code="ORDER_OWNED_BY_CUSTOMER",
            reason="The order exists and belongs to the authenticated customer.",
            observed={"order_id": order_id, "customer_id": customer_id},
        ))

        if not is_write:
            # Ownership settled, and nothing else here governs a read.
            return BusinessRuleResult(checks=checks)

        if order.already_refunded:
            return self._deny(
                checks,
                "ORDER_ALREADY_REFUNDED",
                f"Order {order_id} has already been refunded in full.",
                observed={"order_id": order_id, "refund_status": order.refund_status},
            )
        checks.append(CheckReport(
            check_id="gatekeeper.business_rules.not_refunded",
            name="Refund not already completed",
            status="PASS",
            reason_code="ORDER_NOT_ALREADY_REFUNDED",
            reason="The authoritative order state is not already fully refunded.",
            observed={"order_id": order_id, "refund_status": order.refund_status},
        ))

        if not order.is_refund_eligible:
            return self._deny(
                checks,
                "ORDER_NOT_REFUND_ELIGIBLE",
                order.refund_ineligibility_reason,
                observed={"order_id": order_id, "order_status": order.order_status},
            )
        checks.append(CheckReport(
            check_id="gatekeeper.business_rules.order_eligibility",
            name="Refund eligibility",
            status="PASS",
            reason_code="ORDER_REFUND_ELIGIBLE",
            reason="The authoritative order state allows a refund.",
            observed={"order_id": order_id, "order_status": order.order_status},
        ))

        if not isinstance(amount, (int, float)) or isinstance(amount, bool) or not math.isfinite(float(amount)):
            return self._deny(
                checks,
                "REFUND_AMOUNT_INVALID",
                "The refund amount must be a finite numeric value.",
                observed={"amount": amount},
            )
        amount = float(amount)
        if amount <= 0:
            return self._deny(
                checks,
                "REFUND_AMOUNT_NON_POSITIVE",
                "The refund amount must be greater than zero.",
                observed={"amount": amount},
            )
        if amount > float(order.amount):
            return self._deny(
                checks,
                "REFUND_AMOUNT_EXCEEDS_ORDER",
                f"Requested refund ${amount:,.2f} exceeds order {order_id}'s ${order.amount:,.2f} total.",
                observed={"amount": amount, "order_total": order.amount, "order_id": order_id},
            )

        # The current process_refund tool closes the order as fully refunded.
        # Partial-refund semantics are therefore intentionally not part of the
        # autonomous execution contract in phases 1-3.
        if action == "process_refund" and not math.isclose(amount, float(order.amount), rel_tol=0.0, abs_tol=0.01):
            return self._deny(
                checks,
                "PARTIAL_REFUND_NOT_SUPPORTED",
                f"Direct refund for order {order_id} must equal the authoritative order total of ${order.amount:,.2f}.",
                observed={"amount": amount, "order_total": order.amount, "order_id": order_id},
            )

        checks.append(CheckReport(
            check_id="gatekeeper.business_rules.amount_within_order",
            name="Refund amount within order total",
            status="PASS",
            reason_code="REFUND_AMOUNT_WITHIN_ORDER",
            reason="The requested amount does not exceed the authoritative order total.",
            observed={"amount": amount, "order_total": order.amount, "order_id": order_id},
        ))

        if action == "process_refund" and amount > self._max_refund_amount:
            return BusinessRuleResult(
                verdict="escalate",
                reason_code="EXCEEDS_AGENT_REFUND_LIMIT",
                reason=(
                    f"Requested refund ${amount:,.2f} exceeds this agent's direct refund limit "
                    f"of ${self._max_refund_amount:,.2f}; human review is required."
                ),
                checks=checks + [CheckReport(
                    check_id="gatekeeper.business_rules.agent_refund_limit",
                    name="Agent refund limit",
                    status="ESCALATE",
                    reason_code="EXCEEDS_AGENT_REFUND_LIMIT",
                    reason=(
                        f"Requested refund ${amount:,.2f} exceeds the agent's direct limit "
                        f"of ${self._max_refund_amount:,.2f}."
                    ),
                    observed={
                        "amount": amount,
                        "autonomous_limit": self._max_refund_amount,
                        "remaining_direct_authority": max(0.0, self._max_refund_amount - amount),
                    },
                )],
            )

        if action == "process_refund":
            checks.append(CheckReport(
                check_id="gatekeeper.business_rules.full_refund_only",
                name="Full refund contract",
                status="PASS",
                reason_code="FULL_REFUND_REQUEST",
                reason="The execution tool currently performs a full order refund, so the amount matches the order total.",
                observed={"amount": amount, "order_total": order.amount, "order_id": order_id},
            ))
        if action == "queue_for_approval" and self._hitl is not None:
            pending = self._hitl.pending_for_order(order_id)
            if pending:
                checks.append(CheckReport(
                    check_id="gatekeeper.business_rules.human_review",
                    name="Existing human review",
                    status="PASS",
                    reason_code="REVIEW_ALREADY_PRESENT",
                    reason="A human-review request already exists; the domain handler remains idempotent.",
                    observed={"order_id": order_id, "pending_request": True},
                ))

        return BusinessRuleResult(checks=checks)

    @staticmethod
    def _deny(
        checks: list[CheckReport],
        code: str,
        reason: str,
        *,
        observed: dict[str, Any],
    ) -> BusinessRuleResult:
        checks.append(CheckReport(
            check_id=f"gatekeeper.business_rules.{code.lower()}",
            name="Agent business rule",
            status="FAIL",
            reason_code=code,
            reason=reason,
            observed=observed,
        ))
        return BusinessRuleResult(
            verdict="deny", reason_code=code, reason=reason, checks=checks
        )
