"""Customer Support's authority rule for the Gatekeeper: the refund limit.

This module answers one question about the *agent*, never about the order:
may it move this much money on its own? Whether the order exists, is the
customer's, can be refunded and is being refunded in full are facts about
the *record*. The tool checks those, once, and its write is a
compare-and-swap. They used to be checked here as well, which read the same
order row before every call and made the Gatekeeper a second copy of the
tool rather than the authority it is.

It touches no database. The limit is the persisted agent policy, read once
when the agent is built, and the amount is the one the call proposes. A
proposal below the limit for less than the order's total is still refused,
by the tool, as a partial refund.
"""

from agent_platform.core.primitives.money import Money, money_or_none
from agent_platform.gatekeeper.contracts import BusinessRuleResult, DeterministicBusinessRules
from agent_platform.gatekeeper.types import AuthorityContext, CheckReport
from agent_platform.security.auth.policy import AgentPolicy
from agent_platform.tools.types import ToolCall
from customer_support_agent.config import get_agent_settings

# The budget class this agent's money-moving action belongs to. One name,
# used by the seed data, the Gatekeeper ledger and the rule below.
REFUND_ACTION_CLASS = "refund"
# Why a money-moving action above the automatic limit goes to a person. The
# limit is read from the persisted agent policy; this text reaches the
# trace and the audit trail, never the model.
_OVER_LIMIT = (
    "Order value {amount} exceeds the automatic refund limit of {limit}; "
    "the refund is escalated for human review."
)
_CHECK_ID = "gatekeeper.business_rules.agent_refund_limit"
_CHECK_NAME = "Agent refund limit"


class CustomerSupportBusinessRules(DeterministicBusinessRules):
    """The most this agent may refund without a person, as a Gatekeeper rule."""

    def __init__(
        self, policy: AgentPolicy | None = None, money_actions: frozenset[str] | None = None
    ) -> None:
        # Every action that pays money out of the refund budget. Each is
        # held to the same per-action limit, so a cancellation is not a
        # way around the ceiling a refund is held to.
        self._money_actions = (
            money_actions if money_actions is not None else _refund_budget_actions()
        )
        # None: no ceiling declared for the class, so nothing escalates on amount.
        self._limit = policy.per_action_limit(REFUND_ACTION_CLASS) if policy is not None else None

    def evaluate(self, call: ToolCall, authority: AuthorityContext) -> BusinessRuleResult:
        """Escalate a money-moving call above the limit; pass everything else."""
        action = _bare(call.action)
        amount = money_or_none(call.params.get("amount"))
        if action not in self._money_actions or amount is None:
            return BusinessRuleResult(
                checks=[self._not_applicable(f"'{action}' names no amount of money to move.")]
            )
        if self._limit is None:
            return BusinessRuleResult(
                checks=[self._not_applicable("This agent's policy declares no refund limit.")]
            )
        observed = {"amount": str(amount), "autonomous_limit": str(self._limit)}
        if amount > self._limit:
            return self._escalate(amount, self._limit, observed)
        return BusinessRuleResult(
            checks=[
                CheckReport(
                    check_id=_CHECK_ID,
                    name=_CHECK_NAME,
                    status="PASS",
                    reason_code="WITHIN_AGENT_REFUND_LIMIT",
                    reason="The amount is within what this agent may refund on its own.",
                    observed=observed,
                )
            ]
        )

    def _escalate(
        self, amount: Money, limit: Money, observed: dict[str, str]
    ) -> BusinessRuleResult:
        reason = _OVER_LIMIT.format(amount=amount, limit=limit)
        return BusinessRuleResult(
            verdict="escalate",
            reason_code="EXCEEDS_AGENT_REFUND_LIMIT",
            reason=reason,
            checks=[
                CheckReport(
                    check_id=_CHECK_ID,
                    name=_CHECK_NAME,
                    status="ESCALATE",
                    reason_code="EXCEEDS_AGENT_REFUND_LIMIT",
                    reason=reason,
                    observed=observed,
                )
            ],
        )

    @staticmethod
    def _not_applicable(reason: str) -> CheckReport:
        return CheckReport(
            check_id=_CHECK_ID,
            name=_CHECK_NAME,
            status="SKIPPED",
            reason_code="REFUND_LIMIT_NOT_APPLICABLE",
            reason=reason,
        )


def _bare(action: str) -> str:
    """The tool's own name, without its namespace."""
    return action.rsplit(".", maxsplit=1)[-1]


def _refund_budget_actions() -> frozenset[str]:
    """The tools this agent's config charges to the refund budget."""
    return frozenset(
        tool
        for tool, action_class in get_agent_settings().action_classes.items()
        if action_class == REFUND_ACTION_CLASS
    )
