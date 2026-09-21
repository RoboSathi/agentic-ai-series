"""Deterministic admission control immediately before side effects.

The Gatekeeper does not choose an automation level. That is a design-time
property of the graph. It verifies an exact proposed action against trusted
agent authority, trusted tool metadata, and agent-owned deterministic business
rules, then returns ALLOW, DENY, or ESCALATE.
"""

from typing import Protocol, runtime_checkable

from agentic_commons.audit.ledger import AuditLedger, NullAuditLedger
from agentic_commons.audit.types import AuditEntry
from agentic_commons.config import get_settings
from agentic_commons.gatekeeper.contracts import AllowAllBusinessRules, DeterministicBusinessRules
from agentic_commons.gatekeeper.hierarchy import AuthorityHierarchyResolver
from agentic_commons.gatekeeper.ledger import CumulativeLedger
from agentic_commons.gatekeeper.types import AuthorityContext, AuthorityDecision, CheckReport, LedgerEntry
from agentic_commons.logging import get_logger, instrument, logged
from agentic_commons.logging.trace_context import get_current_trace_id
from agentic_commons.security.auth.auth import Authorizer
from agentic_commons.security.auth.policy import AgentPolicy
from agentic_commons.security.auth.types import AccessDecision, Principal
from agentic_commons.tools.types import ToolCall

_AMOUNT_PARAMS = ("amount", "amount_usd", "value")


@runtime_checkable
class RoleAuthorizer(Protocol):
    """Answers whether this principal may perform this exact action."""

    def authorize(
        self, principal: Principal, action: str, required_permission: str | None = None
    ) -> AccessDecision: ...


@runtime_checkable
class Gatekeeper(Protocol):
    """Mandatory runtime mediation interface before tool execution."""

    async def evaluate(self, call: ToolCall, authority: AuthorityContext) -> AuthorityDecision: ...

    def commit(
        self, call: ToolCall, authority: AuthorityContext, amount: float | None = None
    ) -> None: ...


@instrument("gatekeeper")
class GatekeeperImpl:
    """Generic deterministic reference monitor with agent-owned business rules."""

    _CHECK_ORDER = (
        "agent_enabled",
        "offered",
        "actor",
        "authority",
        "business_rules",
        "cumulative",
    )

    def __init__(
        self,
        *,
        authorizer: RoleAuthorizer,
        policy: AgentPolicy,
        ledger: CumulativeLedger,
        business_rules: DeterministicBusinessRules | None = None,
        audit: AuditLedger | None = None,
        hierarchy: AuthorityHierarchyResolver | None = None,
    ) -> None:
        self._authorizer = authorizer
        self._policy = policy
        self._ledger = ledger
        self._business_rules = business_rules or AllowAllBusinessRules()
        self._audit = audit or NullAuditLedger()
        self._hierarchy = hierarchy or AuthorityHierarchyResolver()
        self._agent_id = policy.agent_id
        self._logger = get_logger("gatekeeper")

    @logged("gatekeeper")
    async def evaluate(self, call: ToolCall, authority: AuthorityContext) -> AuthorityDecision:
        """Verify the exact proposal and return allow, deny, or escalate."""
        context = self._hierarchy.resolve(authority)
        amount = _amount_of(call)
        checks: list[CheckReport] = []
        cumulative_total = 0.0
        threshold = 0.0
        failure: tuple[str, str, str, str] | None = None

        for check_name in self._CHECK_ORDER:
            if failure is not None:
                checks.append(self._skipped(check_name))
                continue
            if check_name == "agent_enabled":
                check, fail = self._eval_agent_enabled()
            elif check_name == "offered":
                check, fail = self._eval_offered(call, context)
            elif check_name == "actor":
                check, fail = self._eval_actor(call, context)
            elif check_name == "authority":
                check, fail = self._eval_authority(call, context)
            elif check_name == "business_rules":
                check, fail = self._eval_business_rules(call, context)
            else:
                check, fail, cumulative_total, threshold = self._eval_cumulative(call, context, amount)
            checks.extend(check)
            if fail is not None:
                failure = fail

        if failure is None:
            decision = AuthorityDecision(
                verdict="allow", action=call.action, actor=context.actor,
                reason="all deterministic Gatekeeper checks passed",
                reason_code="GATEKEEPER_ALLOW", question="complete",
                cumulative_total=cumulative_total, threshold=threshold,
                authority_source=context.granted_by, authority_rank=self._policy.authority_rank,
                required_authority_rank=call.requires_authority_rank,
                max_granting_rank=self._hierarchy.max_granting_rank, checks=checks,
            )
        else:
            question, verdict, reason_code, reason = failure
            decision = AuthorityDecision(
                verdict=verdict, action=call.action, actor=context.actor,
                reason=reason, reason_code=reason_code, question=question,
                cumulative_total=cumulative_total, threshold=threshold,
                authority_source=context.granted_by, authority_rank=self._policy.authority_rank,
                required_authority_rank=call.requires_authority_rank,
                max_granting_rank=self._hierarchy.max_granting_rank, checks=checks,
            )

        self._record(call, context, decision)
        self._logger.info(
            "Gatekeeper decision", agent_id=self._agent_id, policy_role=self._policy.role,
            policy_updated_at=self._policy.updated_at, action=call.action, actor=context.actor,
            verdict=decision.verdict, reason_code=decision.reason_code, question=decision.question,
            reason=decision.reason, checks=[_compact_check(c) for c in decision.checks],
        )
        return decision

    def commit(self, call: ToolCall, authority: AuthorityContext, amount: float | None = None) -> None:
        """Record an allowed action's actual financial consumption after execution."""
        spent = amount if amount is not None else _amount_of(call)
        if spent <= 0 or not self._ledger.is_budgeted(call.action):
            return
        self._ledger.record_and_check(
            LedgerEntry(
                agent_id=self._agent_id,
                ledger_key=_ledger_key(authority),
                action_class=self._ledger.class_of(call.action),
                actor=authority.actor, amount=spent,
                trace_id=get_current_trace_id() or "",
            )
        )

    def _eval_agent_enabled(self):
        passed = self._policy.enabled
        check = CheckReport(
            check_id="gatekeeper.agent_enabled", name="Agent enabled",
            status="PASS" if passed else "FAIL",
            reason_code="AGENT_ENABLED" if passed else "AGENT_DISABLED",
            reason=("Persisted agent policy is enabled." if passed else "Persisted agent policy is disabled."),
            observed={"agent_id": self._agent_id, "enabled": passed},
        )
        return [check], None if passed else ("agent_enabled", "deny", "AGENT_DISABLED", check.reason)

    def _eval_offered(self, call: ToolCall, context: AuthorityContext):
        offered = context.offered_actions
        passed = offered is None or call.action in offered
        check = CheckReport(
            check_id="gatekeeper.offered_action", name="Offered action",
            status="PASS" if passed else "FAIL",
            reason_code="ACTION_OFFERED" if passed else "ACTION_NOT_OFFERED",
            reason=("The exact action was offered to the model." if passed else f"'{call.action}' was not offered."),
            observed={"action": call.action},
            evidence={"offered_actions": sorted(offered or [])},
        )
        return [check], None if passed else ("offered", "deny", "ACTION_NOT_OFFERED", check.reason)

    def _eval_actor(self, call: ToolCall, context: AuthorityContext):
        outcome = self._authorizer.authorize(
            Principal(actor=context.actor), call.action, call.required_permission
        )
        check = CheckReport(
            check_id="gatekeeper.actor_authorization", name="Agent authorization",
            status="PASS" if outcome.allowed else "FAIL",
            reason_code="ACTOR_AUTHORIZED" if outcome.allowed else "ACTOR_NOT_AUTHORIZED",
            reason=outcome.reason,
            observed={
                "actor": context.actor, "action": call.action,
                "role": outcome.role, "required_permission": outcome.permission,
            },
            evidence={"authorization_allowed": outcome.allowed},
        )
        return [check], None if outcome.allowed else ("actor", "deny", "ACTOR_NOT_AUTHORIZED", outcome.reason)

    def _eval_authority(self, call: ToolCall, context: AuthorityContext):
        may_grant, hierarchy_reason = self._hierarchy.may_authorize(context)
        required_rank = call.requires_authority_rank
        agent_rank = self._policy.authority_rank
        source_rank = self._hierarchy.rank_of(context.granted_by)
        passed = may_grant and agent_rank <= required_rank and source_rank <= required_rank
        if passed:
            code = "AUTHORITY_GRANTED"
            reason = f"Agent rank {agent_rank} and source '{context.granted_by}' rank {source_rank} satisfy required rank {required_rank}."
        elif not may_grant:
            code, reason = "AUTHORITY_SOURCE_CANNOT_GRANT", hierarchy_reason
        elif agent_rank > required_rank:
            code, reason = "AGENT_AUTHORITY_RANK_INSUFFICIENT", f"Agent rank {agent_rank} is insufficient for required rank {required_rank}."
        else:
            code, reason = "AUTHORITY_SOURCE_INSUFFICIENT", f"Source rank {source_rank} is insufficient for required rank {required_rank}."
        check = CheckReport(
            check_id="gatekeeper.authority_hierarchy", name="Authority hierarchy",
            status="PASS" if passed else "FAIL", reason_code=code, reason=reason,
            observed={"agent_id": self._agent_id, "agent_rank": agent_rank, "granted_by": context.granted_by, "granted_by_rank": source_rank, "required_rank": required_rank},
            evidence={"hierarchy": self._hierarchy.hierarchy},
        )
        return [check], None if passed else ("authority", "deny", code, reason)

    def _eval_business_rules(self, call: ToolCall, context: AuthorityContext):
        result = self._business_rules.evaluate(call, context)
        if result.verdict == "allow":
            return result.checks, None
        failed = next((c for c in result.checks if c.status in {"FAIL", "ESCALATE"}), None)
        reason = result.reason or (failed.reason if failed else result.reason_code)
        verdict = result.verdict
        check_status = "ESCALATE" if verdict == "escalate" else "FAIL"
        checks = result.checks or [CheckReport(
            check_id="gatekeeper.business_rules", name="Agent business rules",
            status=check_status, reason_code=result.reason_code, reason=reason,
        )]
        return checks, ("business_rules", verdict, result.reason_code, reason)

    def _eval_cumulative(self, call: ToolCall, context: AuthorityContext, amount: float):
        if not self._ledger.enabled or not self._ledger.is_budgeted(call.action):
            code = "CUMULATIVE_DISABLED" if not self._ledger.enabled else "CUMULATIVE_NOT_APPLICABLE"
            return [CheckReport(
                check_id="gatekeeper.cumulative_limit", name="Cumulative budget", status="SKIPPED",
                reason_code=code, reason=("Cumulative accounting is disabled." if not self._ledger.enabled else "This action is not mapped to a financial budget."),
            )], None, 0.0, 0.0
        action_class = self._ledger.class_of(call.action)
        if not self._ledger.threshold_configured(action_class):
            reason = f"Cumulative budget class '{action_class}' has no finite policy threshold."
            return [CheckReport(
                check_id="gatekeeper.cumulative_limit", name="Cumulative budget", status="FAIL",
                reason_code="CUMULATIVE_POLICY_MISCONFIGURED", reason=reason, observed={"action_class": action_class},
            )], ("cumulative", "deny", "CUMULATIVE_POLICY_MISCONFIGURED", reason), 0.0, 0.0
        ledger_key = _ledger_key(context)
        previous = self._ledger.window_total(ledger_key, action_class)
        state = self._ledger.state(ledger_key, action_class, pending=amount)
        passed = not state.exceeded
        check = CheckReport(
            check_id="gatekeeper.cumulative_limit", name="Cumulative budget",
            status="PASS" if passed else "FAIL",
            reason_code="CUMULATIVE_WITHIN_LIMIT" if passed else "CUMULATIVE_LIMIT_EXCEEDED",
            reason=(f"Projected {action_class} spend ${state.window_total:,.2f} is within the ${state.threshold:,.2f} rolling limit." if passed else f"Projected {action_class} spend ${state.window_total:,.2f} exceeds the ${state.threshold:,.2f} rolling limit."),
            observed={"action_class": action_class, "ledger_key": ledger_key, "previous": previous, "proposed": amount, "projected": state.window_total, "threshold": state.threshold, "remaining_before": max(0.0, state.threshold - previous), "remaining_after": max(0.0, state.threshold - state.window_total), "entry_count": state.entry_count},
        )
        return [check], None if passed else ("cumulative", "deny", "CUMULATIVE_LIMIT_EXCEEDED", check.reason), state.window_total, state.threshold

    @staticmethod
    def _skipped(check_name: str) -> CheckReport:
        return CheckReport(
            check_id=f"gatekeeper.{check_name}", name=check_name.replace("_", " ").title(),
            status="SKIPPED", reason_code="SHORT_CIRCUITED",
            reason="Not evaluated because an earlier deterministic Gatekeeper check stopped the proposal.",
        )

    def _record(self, call: ToolCall, context: AuthorityContext, decision: AuthorityDecision) -> None:
        llm_settings = get_settings().llm
        model = llm_settings.providers[llm_settings.provider].model
        self._audit.record(
            AuditEntry(
                trace_id=get_current_trace_id() or "", actor=context.actor, action=call.action,
                authority_context={**context.model_dump(mode="json"), "agent_id": self._agent_id, "policy_role": self._policy.role, "policy_authority_rank": self._policy.authority_rank, "policy_updated_at": self._policy.updated_at},
                model_version=model, decision=decision.verdict, rationale=decision.reason,
            )
        )


def _compact_check(check: CheckReport) -> dict[str, object]:
    payload: dict[str, object] = {"name": check.name, "status": check.status, "reason_code": check.reason_code, "reason": check.reason}
    if check.observed:
        payload["observed"] = check.observed
    if check.evidence:
        payload["evidence"] = check.evidence
    return payload


def _amount_of(call: ToolCall) -> float:
    for name in _AMOUNT_PARAMS:
        value = call.params.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return 0.0


def _ledger_key(context: AuthorityContext) -> str:
    return context.cumulative_ledger_key or context.actor
