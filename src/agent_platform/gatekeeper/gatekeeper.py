"""Deterministic admission control immediately before side effects.

The Gatekeeper decides one thing: whether the agent has *authority* to run
the exact call the model proposed. It does not ask whether the action is
correct for the record (the tool does) or whether the input is safe
(screening does). It reads trusted agent policy, trusted tool metadata and
the agent's own authority rules, returns ALLOW, DENY or ESCALATE, and names
the action's standing: VOID, RATIFIABLE, VOIDABLE or VALID.
"""

from typing import Protocol, runtime_checkable

from agent_platform.audit.ledger import AuditLedger, NullAuditLedger
from agent_platform.audit.types import AuditEntry
from agent_platform.config import get_settings
from agent_platform.gatekeeper.contracts import AllowAllBusinessRules, DeterministicBusinessRules
from agent_platform.gatekeeper.hierarchy import AuthorityHierarchyResolver
from agent_platform.gatekeeper.ledger import CumulativeLedger
from agent_platform.gatekeeper.types import (
    AuthorityContext,
    AuthorityDecision,
    CheckReport,
    CheckStatus,
    LedgerEntry,
    Verdict,
)
from agent_platform.logging import get_logger, instrument, logged
from agent_platform.logging.trace_context import get_current_trace_id
from agent_platform.primitives.money import Money, money_or_none
from agent_platform.security.auth.policy import AgentPolicy
from agent_platform.security.auth.types import AccessDecision, Principal
from agent_platform.tools.types import ToolCall

_AMOUNT_PARAMS = ("amount", "amount_usd", "value")

# One check's failure: (question, verdict, reason_code, reason), or None.
_Failure = tuple[str, Verdict, str, str] | None
# What every `_eval_*` step returns: the checks it ran, and its failure.
_Evaluation = tuple[list[CheckReport], _Failure]


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
        self, call: ToolCall, authority: AuthorityContext, amount: Money | None = None
    ) -> None: ...


@instrument("gatekeeper")
class GatekeeperImpl:
    """Generic deterministic reference monitor with agent-owned business rules."""

    # `mandate` sits immediately after `offered` on purpose. `offered`
    # asks whether the menu was respected; `mandate` asks whether anybody
    # asked for this. Those had been the same question, which made tool
    # exposure and authorization the same act.
    _CHECK_ORDER = (
        "agent_enabled",
        "offered",
        "mandate",
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
        failure: _Failure = None

        for check_name in self._CHECK_ORDER:
            # A deployment without cumulative accounting has no such check
            # to report, not a check that ran and was skipped every time.
            if check_name == "cumulative" and not self._ledger.enabled:
                continue
            if failure is not None:
                checks.append(self._skipped(check_name))
                continue
            if check_name == "agent_enabled":
                check, fail = self._eval_agent_enabled()
            elif check_name == "offered":
                check, fail = self._eval_offered(call, context)
            elif check_name == "mandate":
                check, fail = self._eval_mandate(call, context, amount)
            elif check_name == "actor":
                check, fail = self._eval_actor(call, context)
            elif check_name == "authority":
                check, fail = self._eval_authority(call, context)
            elif check_name == "business_rules":
                check, fail = self._eval_business_rules(call, context)
            else:
                check, fail = self._eval_cumulative(call, context, amount)
            checks.extend(check)
            if fail is not None:
                failure = fail

        if failure is None:
            decision = AuthorityDecision(
                verdict="allow",
                action=call.action,
                actor=context.actor,
                reason="all deterministic Gatekeeper checks passed",
                reason_code="GATEKEEPER_ALLOW",
                question="complete",
                checks=checks,
                # Allowed, and still undoable for a while: the principal keeps
                # the right to void it until its window closes.
                standing="VOIDABLE" if call.voidable else "VALID",
            )
        else:
            question, verdict, reason_code, reason = failure
            decision = AuthorityDecision(
                verdict=verdict,
                action=call.action,
                actor=context.actor,
                reason=reason,
                reason_code=reason_code,
                question=question,
                checks=checks,
            )

        self._record(call, context, decision)
        self._logger.info(
            "Gatekeeper decision",
            agent_id=self._agent_id,
            policy_role=self._policy.role,
            policy_updated_at=self._policy.updated_at,
            action=call.action,
            actor=context.actor,
            verdict=decision.verdict,
            standing=decision.standing,
            reason_code=decision.reason_code,
            question=decision.question,
            reason=decision.reason,
            checks=[_compact_check(c) for c in decision.checks],
        )
        return decision

    def commit(
        self, call: ToolCall, authority: AuthorityContext, amount: Money | None = None
    ) -> None:
        """Record an allowed action's actual financial consumption after execution."""
        spent = amount if amount is not None else _amount_of(call)
        if not spent or not self._ledger.is_budgeted(call.action):
            return
        self._ledger.record_and_check(
            LedgerEntry(
                agent_id=self._agent_id,
                ledger_key=_ledger_key(authority),
                action_class=self._ledger.class_of(call.action),
                actor=authority.actor,
                amount=spent,
                trace_id=get_current_trace_id() or "",
            )
        )

    def _eval_agent_enabled(self) -> _Evaluation:
        passed = self._policy.enabled
        check = CheckReport(
            check_id="gatekeeper.agent_enabled",
            name="Agent enabled",
            status="PASS" if passed else "FAIL",
            reason_code="AGENT_ENABLED" if passed else "AGENT_DISABLED",
            reason=(
                "Persisted agent policy is enabled."
                if passed
                else "Persisted agent policy is disabled."
            ),
            observed={"agent_id": self._agent_id, "enabled": passed},
        )
        return [check], None if passed else (
            "agent_enabled",
            "deny",
            "AGENT_DISABLED",
            check.reason,
        )

    def _eval_offered(self, call: ToolCall, context: AuthorityContext) -> _Evaluation:
        offered = context.offered_actions
        passed = offered is None or call.action in offered
        check = CheckReport(
            check_id="gatekeeper.offered_action",
            name="Offered action",
            status="PASS" if passed else "FAIL",
            reason_code="ACTION_OFFERED" if passed else "ACTION_NOT_OFFERED",
            reason=(
                "The exact action was offered to the model."
                if passed
                else f"'{call.action}' was not offered."
            ),
            observed={"action": call.action},
            evidence={"offered_actions": sorted(offered or [])},
        )
        return [check], None if passed else ("offered", "deny", "ACTION_NOT_OFFERED", check.reason)

    def _eval_mandate(
        self, call: ToolCall, context: AuthorityContext, amount: Money | None
    ) -> _Evaluation:
        """Whether this proposal acts on a record the turn established.

        Only side-effecting calls are scoped. A read changes nothing, so
        what it names is a data-access question the business rules settle.

        This check does **not** ask whether the person requested the
        action — that is language work, and an authorization path that
        does it with a keyword list gets it wrong in both directions.
        Whether the action may proceed at all is the agent's deterministic
        business rules; this is the narrow question of *which* record.

        A `None` set means no issuer is wired, and the check says so
        rather than inventing a verdict. An *empty* set is a different
        statement: the issuer looked and found no established target, so
        a write has nothing it may legitimately act on.
        """
        if not call.side_effecting:
            return [
                CheckReport(
                    check_id="gatekeeper.mandate",
                    name="Target in scope",
                    status="SKIPPED",
                    reason_code="MANDATE_NOT_REQUIRED",
                    reason="A read changes nothing, so it needs no write scope.",
                    observed={"action": call.action},
                )
            ], None
        if context.mandates is None:
            return [
                CheckReport(
                    check_id="gatekeeper.mandate",
                    name="Target in scope",
                    status="SKIPPED",
                    reason_code="MANDATE_NOT_CONFIGURED",
                    reason="This deployment scopes no writes, so the target was not checked.",
                    observed={"action": call.action},
                )
            ], None
        target = self._target_of(call)
        mandate = context.mandates.covering(call, target, amount, context.turn)
        passed = mandate is not None
        check = CheckReport(
            check_id="gatekeeper.mandate",
            name="Target in scope",
            status="PASS" if passed else "FAIL",
            reason_code="TARGET_IN_SCOPE" if passed else "TARGET_NOT_IN_SCOPE",
            reason=(
                f"{target} is the record this conversation established."
                if passed
                else f"'{_bare(call.action)}' names {target}, which is not a record this "
                f"conversation established" + (f" for {amount}" if amount else "") + "."
            ),
            observed={
                "action": call.action,
                "target": target,
                "amount": str(amount) if amount else None,
                "turn": context.turn,
            },
            evidence={
                "mandates": [
                    {
                        "action": m.action,
                        "target": m.target,
                        "max_amount": str(m.max_amount)
                        if m.max_amount is not None
                        else "unbounded",
                        "granted_by": m.granted_by,
                        "evidence": m.evidence,
                    }
                    for m in context.mandates.mandates
                ]
            },
        )
        return [check], None if passed else ("mandate", "deny", "TARGET_NOT_IN_SCOPE", check.reason)

    def _target_of(self, call: ToolCall) -> str:
        """What this call acts on, as a mandate names it.

        Built from the call's own parameters using the agent's configured
        target key, so the platform needs no opinion about what a target
        is called.
        """
        key = get_settings().gatekeeper.mandate.target_param
        value = call.params.get(key)
        return f"{key.removesuffix('_id')}:{value}" if value is not None else ""

    def _eval_actor(self, call: ToolCall, context: AuthorityContext) -> _Evaluation:
        outcome = self._authorizer.authorize(
            Principal(actor=context.actor), call.action, call.required_permission
        )
        check = CheckReport(
            check_id="gatekeeper.actor_authorization",
            name="Agent authorization",
            status="PASS" if outcome.allowed else "FAIL",
            reason_code="ACTOR_AUTHORIZED" if outcome.allowed else "ACTOR_NOT_AUTHORIZED",
            reason=outcome.reason,
            observed={
                "actor": context.actor,
                "action": call.action,
                "role": outcome.role,
                "required_permission": outcome.permission,
            },
            evidence={"authorization_allowed": outcome.allowed},
        )
        return [check], None if outcome.allowed else (
            "actor",
            "deny",
            "ACTOR_NOT_AUTHORIZED",
            outcome.reason,
        )

    def _eval_authority(self, call: ToolCall, context: AuthorityContext) -> _Evaluation:
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
            code, reason = (
                "AGENT_AUTHORITY_RANK_INSUFFICIENT",
                f"Agent rank {agent_rank} is insufficient for required rank {required_rank}.",
            )
        else:
            code, reason = (
                "AUTHORITY_SOURCE_INSUFFICIENT",
                f"Source rank {source_rank} is insufficient for required rank {required_rank}.",
            )
        check = CheckReport(
            check_id="gatekeeper.authority_hierarchy",
            name="Authority hierarchy",
            status="PASS" if passed else "FAIL",
            reason_code=code,
            reason=reason,
            observed={
                "agent_id": self._agent_id,
                "agent_rank": agent_rank,
                "granted_by": context.granted_by,
                "granted_by_rank": source_rank,
                "required_rank": required_rank,
            },
            evidence={"hierarchy": self._hierarchy.hierarchy},
        )
        return [check], None if passed else ("authority", "deny", code, reason)

    def _eval_business_rules(self, call: ToolCall, context: AuthorityContext) -> _Evaluation:
        result = self._business_rules.evaluate(call, context)
        if result.verdict == "allow":
            return result.checks, None
        failed = next((c for c in result.checks if c.status in {"FAIL", "ESCALATE"}), None)
        reason = result.reason or (failed.reason if failed else result.reason_code)
        verdict = result.verdict
        check_status: CheckStatus = "ESCALATE" if verdict == "escalate" else "FAIL"
        checks = result.checks or [
            CheckReport(
                check_id="gatekeeper.business_rules",
                name="Agent business rules",
                status=check_status,
                reason_code=result.reason_code,
                reason=reason,
            )
        ]
        return checks, ("business_rules", verdict, result.reason_code, reason)

    def _eval_cumulative(
        self, call: ToolCall, context: AuthorityContext, amount: Money | None
    ) -> _Evaluation:
        if not self._ledger.enabled or not self._ledger.is_budgeted(call.action):
            code = (
                "CUMULATIVE_DISABLED" if not self._ledger.enabled else "CUMULATIVE_NOT_APPLICABLE"
            )
            return [
                CheckReport(
                    check_id="gatekeeper.cumulative_limit",
                    name="Cumulative budget",
                    status="SKIPPED",
                    reason_code=code,
                    reason=(
                        "Cumulative accounting is disabled."
                        if not self._ledger.enabled
                        else "This action is not mapped to a financial budget."
                    ),
                )
            ], None
        action_class = self._ledger.class_of(call.action)
        if not self._ledger.threshold_configured(action_class):
            reason = f"Cumulative budget class '{action_class}' has no finite policy threshold."
            return (
                [
                    CheckReport(
                        check_id="gatekeeper.cumulative_limit",
                        name="Cumulative budget",
                        status="FAIL",
                        reason_code="CUMULATIVE_POLICY_MISCONFIGURED",
                        reason=reason,
                        observed={"action_class": action_class},
                    )
                ],
                ("cumulative", "deny", "CUMULATIVE_POLICY_MISCONFIGURED", reason),
            )
        ledger_key = _ledger_key(context)
        previous = self._ledger.window_total(ledger_key, action_class)
        state = self._ledger.state(ledger_key, action_class, pending=amount)
        passed = not state.exceeded
        check = CheckReport(
            check_id="gatekeeper.cumulative_limit",
            name="Cumulative budget",
            status="PASS" if passed else "FAIL",
            reason_code="CUMULATIVE_WITHIN_LIMIT" if passed else "CUMULATIVE_LIMIT_EXCEEDED",
            reason=(
                f"Projected {action_class} spend {state.window_total} is within the {state.threshold} rolling limit."
                if passed
                else f"Projected {action_class} spend {state.window_total} exceeds the {state.threshold} rolling limit."
            ),
            observed={
                "action_class": action_class,
                "ledger_key": ledger_key,
                "previous": str(previous),
                "proposed": str(amount or Money.zero()),
                "projected": str(state.window_total),
                "threshold": str(state.threshold),
                "remaining_after": str(state.remaining),
                "entry_count": state.entry_count,
            },
        )
        return [check], None if passed else (
            "cumulative",
            "deny",
            "CUMULATIVE_LIMIT_EXCEEDED",
            check.reason,
        )

    @staticmethod
    def _skipped(check_name: str) -> CheckReport:
        return CheckReport(
            check_id=f"gatekeeper.{check_name}",
            name=check_name.replace("_", " ").title(),
            status="SKIPPED",
            reason_code="SHORT_CIRCUITED",
            reason="Not evaluated because an earlier deterministic Gatekeeper check stopped the proposal.",
        )

    def _record(
        self, call: ToolCall, context: AuthorityContext, decision: AuthorityDecision
    ) -> None:
        llm_settings = get_settings().llm
        model = llm_settings.providers[llm_settings.provider].model
        self._audit.record(
            AuditEntry(
                trace_id=get_current_trace_id() or "",
                actor=context.actor,
                action=call.action,
                authority_context={
                    **context.model_dump(mode="json"),
                    "agent_id": self._agent_id,
                    "policy_role": self._policy.role,
                    "policy_authority_rank": self._policy.authority_rank,
                    "policy_updated_at": self._policy.updated_at,
                    "standing": decision.standing,
                },
                model_version=model,
                decision=decision.verdict,
                rationale=decision.reason,
            )
        )


def _compact_check(check: CheckReport) -> dict[str, object]:
    payload: dict[str, object] = {
        "name": check.name,
        "status": check.status,
        "reason_code": check.reason_code,
        "reason": check.reason,
    }
    if check.observed:
        payload["observed"] = check.observed
    if check.evidence:
        payload["evidence"] = check.evidence
    return payload


def _bare(action: str) -> str:
    """A tool's own name, without its namespace."""
    return action.rsplit(".", maxsplit=1)[-1]


def _amount_of(call: ToolCall) -> Money | None:
    """The amount a call proposes, exactly — or None if it names no valid amount.

    An invalid amount (negative, more than two decimals) is None here and
    refused by the agent's own business rules, which say why.
    """
    for name in _AMOUNT_PARAMS:
        value = call.params.get(name)
        if value is not None:
            return money_or_none(value)
    return None


def _ledger_key(context: AuthorityContext) -> str:
    return context.cumulative_ledger_key or context.actor
