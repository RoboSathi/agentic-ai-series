"""Customer Support's deterministic business rules for the Gatekeeper."""

from typing import Any

from agent_platform.gatekeeper.contracts import BusinessRuleResult, DeterministicBusinessRules
from agent_platform.gatekeeper.types import AuthorityContext, CheckReport
from agent_platform.security.auth.policy import AgentPolicy
from agent_platform.tools.types import ToolCall
from customer_support_agent.config import get_agent_settings
from customer_support_agent.domain.cancellation_policy import CancellationPolicy
from customer_support_agent.domain.refund_policy import RefundPolicy, RefundValidationResult
from customer_support_agent.persistence.repositories import HitlQueueRepository, OrdersRepository

# The budget class this agent's money-moving action belongs to. One name,
# used by the seed data, the Gatekeeper ledger and the rule below.
REFUND_ACTION_CLASS = "refund"
# Cancelling an unshipped order refunds it, so it is a write on an order too.
_CANCEL_ACTION = "cancel_order"
# Why a money-moving action above the automatic limit goes to a person. The
# limit is read from the persisted agent policy; this text reaches the
# trace and the audit trail, never the model.
_OVER_LIMIT = (
    "Order value {amount} exceeds the automatic refund limit of {limit}; "
    "the refund is escalated for human review."
)


class CustomerSupportBusinessRules(DeterministicBusinessRules):
    """Authoritative order/refund checks that must run immediately before writes."""

    def __init__(
        self,
        orders: OrdersRepository,
        hitl: HitlQueueRepository | None = None,
        policy: AgentPolicy | None = None,
        refund_policy: RefundPolicy | None = None,
        money_actions: frozenset[str] | None = None,
    ) -> None:
        self._orders = orders
        # Every action that pays money out of the refund budget. Each is
        # held to the same per-action limit, so a cancellation is not a
        # way around the ceiling a refund is held to.
        self._money_actions = (
            money_actions if money_actions is not None else _refund_budget_actions()
        )
        self._cancellation_policy = CancellationPolicy(orders)
        self._hitl = hitl
        # This agent's own action class. The platform stores budgets keyed
        # by class and does not know what "refund" means; naming it is this
        # module's job, because this is the module that has refunds.
        # None: no ceiling declared for the class.
        self._max_refund_amount = (
            policy.per_action_limit(REFUND_ACTION_CLASS) if policy is not None else None
        )
        self._refund_policy = refund_policy or RefundPolicy(
            orders, direct_limit=self._max_refund_amount
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
        is_write = action in {"process_refund", "queue_for_approval", _CANCEL_ACTION}
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
        checks.append(
            CheckReport(
                check_id="gatekeeper.business_rules.customer",
                name="Authenticated customer",
                status="PASS",
                reason_code="CUSTOMER_AUTHENTICATED",
                reason="The action is scoped to an authenticated customer.",
                observed={"customer_id": customer_id},
            )
        )

        if not isinstance(order_id, int):
            return self._deny(
                checks,
                "ORDER_ID_REQUIRED",
                f"A concrete order id is required for '{action}'.",
                observed={"order_id": order_id},
            )

        if not is_write:
            order = self._orders.get_for_customer(order_id, customer_id)
            if order is None:
                return self._deny(
                    checks,
                    "ORDER_NOT_FOUND_FOR_CUSTOMER",
                    f"Order {order_id} was not found on the authenticated customer's account.",
                    observed={"order_id": order_id, "customer_id": customer_id},
                )
            checks.append(
                CheckReport(
                    check_id="gatekeeper.business_rules.order_ownership",
                    name="Order ownership",
                    status="PASS",
                    reason_code="ORDER_OWNED_BY_CUSTOMER",
                    reason="The order exists and belongs to the authenticated customer.",
                    observed={"order_id": order_id, "customer_id": customer_id},
                )
            )
            return BusinessRuleResult(checks=checks)

        validation = self._validate_write(action, order_id, customer_id, amount)
        order = validation.order
        if order is None:
            return self._deny(
                checks,
                validation.reason_code,
                validation.reason,
                observed={"order_id": order_id, "customer_id": customer_id},
            )
        checks.append(
            CheckReport(
                check_id="gatekeeper.business_rules.order_ownership",
                name="Order ownership",
                status="PASS",
                reason_code="ORDER_OWNED_BY_CUSTOMER",
                reason="The order exists and belongs to the authenticated customer.",
                observed={"order_id": order_id, "customer_id": customer_id},
            )
        )

        if not validation.allowed:
            return self._deny(
                checks,
                validation.reason_code,
                validation.reason,
                observed={
                    "order_id": order_id,
                    "customer_id": customer_id,
                    "amount": str(validation.amount) if validation.amount else str(amount),
                    "order_status": order.order_status,
                    "refund_status": order.refund_status,
                },
            )

        normalized_amount = validation.amount
        assert normalized_amount is not None
        checks.extend(
            [
                CheckReport(
                    check_id="gatekeeper.business_rules.not_refunded",
                    name="Refund not already completed",
                    status="PASS",
                    reason_code="ORDER_NOT_ALREADY_REFUNDED",
                    reason="The authoritative order state is not already fully refunded.",
                    observed={"order_id": order_id, "refund_status": order.refund_status},
                ),
                CheckReport(
                    check_id="gatekeeper.business_rules.order_eligibility",
                    name="Refund eligibility",
                    status="PASS",
                    reason_code="ORDER_REFUND_ELIGIBLE",
                    reason="The authoritative order state allows a refund.",
                    observed={"order_id": order_id, "order_status": order.order_status},
                ),
                CheckReport(
                    check_id="gatekeeper.business_rules.amount_within_order",
                    name="Refund amount within order total",
                    status="PASS",
                    reason_code="REFUND_AMOUNT_WITHIN_ORDER",
                    reason="The requested amount is the full authoritative order total.",
                    observed={
                        "amount": str(normalized_amount),
                        "order_total": str(order.amount),
                        "order_id": order_id,
                    },
                ),
            ]
        )

        limit = self._max_refund_amount
        if action in self._money_actions and limit is not None and normalized_amount > limit:
            return BusinessRuleResult(
                verdict="escalate",
                reason_code="EXCEEDS_AGENT_REFUND_LIMIT",
                reason=_OVER_LIMIT.format(amount=normalized_amount, limit=limit),
                checks=checks
                + [
                    CheckReport(
                        check_id="gatekeeper.business_rules.agent_refund_limit",
                        name="Agent refund limit",
                        status="ESCALATE",
                        reason_code="EXCEEDS_AGENT_REFUND_LIMIT",
                        reason=_OVER_LIMIT.format(amount=normalized_amount, limit=limit),
                        observed={
                            "amount": str(normalized_amount),
                            "autonomous_limit": str(limit),
                        },
                    )
                ],
            )

        if action == "queue_for_approval" and self._hitl is not None:
            pending = self._hitl.pending_for_order(order_id)
            if pending:
                checks.append(
                    CheckReport(
                        check_id="gatekeeper.business_rules.human_review",
                        name="Existing human review",
                        status="PASS",
                        reason_code="REVIEW_ALREADY_PRESENT",
                        reason="A human-review request already exists; the domain handler remains idempotent.",
                        observed={"order_id": order_id, "pending_request": True},
                    )
                )

        return BusinessRuleResult(checks=checks)

    def _validate_write(
        self, action: str, order_id: int, customer_id: int, amount: Any
    ) -> RefundValidationResult:
        """The domain rules for this write: a cancellation's, or a refund's."""
        if action == _CANCEL_ACTION:
            return self._cancellation_policy.validate(order_id, customer_id, amount)
        return self._refund_policy.validate(
            order_id=order_id,
            customer_id=customer_id,
            amount=amount,
            action=action,  # type: ignore[arg-type]
        )

    @staticmethod
    def _deny(
        checks: list[CheckReport],
        code: str,
        reason: str,
        *,
        observed: dict[str, Any],
    ) -> BusinessRuleResult:
        checks.append(
            CheckReport(
                check_id=f"gatekeeper.business_rules.{code.lower()}",
                name="Agent business rule",
                status="FAIL",
                reason_code=code,
                reason=reason,
                observed=observed,
            )
        )
        return BusinessRuleResult(verdict="deny", reason_code=code, reason=reason, checks=checks)


def _refund_budget_actions() -> frozenset[str]:
    """The tools this agent's config charges to the refund budget."""
    return frozenset(
        tool
        for tool, action_class in get_agent_settings().action_classes.items()
        if action_class == REFUND_ACTION_CLASS
    )
