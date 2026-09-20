"""The Gatekeeper — the final deterministic admission check before execution.

The Gatekeeper answers one question:

    "Is this exact, already-validated tool call admissible to execute now?"

It evaluates deterministic controls in a deliberate order and short-circuits
on the first blocking result. Every unexecuted downstream check is explicitly
reported as ``SKIPPED`` so the TUI and logs never imply a check passed when it
was not actually run.

Tier 2 remains advisory only. A successful Gatekeeper decision is converted to
``ApprovedToolCall`` by the caller and only that contract may cross into the
common Executor.
"""

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from agentic_commons.audit.ledger import AuditLedger, NullAuditLedger
from agentic_commons.audit.types import AuditEntry
from agentic_commons.basic_sanity import BasicSanityChecker, BasicSanityRequest
from agentic_commons.config import get_settings
from agentic_commons.gatekeeper.automation import AutomationLevelPolicy
from agentic_commons.gatekeeper.hierarchy import AuthorityHierarchyResolver
from agentic_commons.gatekeeper.ledger import CumulativeLedger
from agentic_commons.gatekeeper.surfaces import SurfaceClassifier, ThreeGatesEvaluator
from agentic_commons.gatekeeper.tier2 import NullTier2Advisor, Tier2Advisor
from agentic_commons.gatekeeper.types import (
    AdvisoryFlag,
    AuthorityContext,
    AuthorityDecision,
    AutomationAssessment,
    CheckReport,
    GateEvaluation,
    LedgerEntry,
)
from agentic_commons.logging import get_logger, instrument, logged
from agentic_commons.logging.trace_context import get_current_trace_id
from agentic_commons.security.auth.types import AccessDecision, Principal
from agentic_commons.tools.types import ToolCall

_AMOUNT_PARAMS = ("amount", "amount_usd", "value")


def _compact_check(check: CheckReport) -> dict[str, object]:
    """Serialize only operator-useful check data for the canonical Gatekeeper log."""
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


@runtime_checkable
class RoleAuthorizer(Protocol):
    """Answers "may this principal perform this action?" (RBAC)."""

    def authorize(self, principal: Principal, action: str) -> AccessDecision: ...


@runtime_checkable
class Gatekeeper(Protocol):
    """The mediation point every execution path passes through."""

    async def evaluate(self, call: ToolCall, authority: AuthorityContext) -> AuthorityDecision: ...


@instrument("gatekeeper")
class GatekeeperImpl:
    """Deterministic reference monitor over the complete execution envelope."""

    _CHECK_ORDER: tuple[str, ...] = (
        "offered",
        "surface",
        "actor",
        "authority",
        "basic_sanity",
        "cumulative",
        "three_gates",
        "automation",
    )

    def __init__(
        self,
        authorizer: RoleAuthorizer | None = None,
        ledger: CumulativeLedger | None = None,
        hierarchy: AuthorityHierarchyResolver | None = None,
        surfaces: SurfaceClassifier | None = None,
        automation: AutomationLevelPolicy | None = None,
        tier2: Tier2Advisor | None = None,
        audit: AuditLedger | None = None,
        basic_sanity: BasicSanityChecker | None = None,
    ) -> None:
        self._authorizer = authorizer
        self._ledger = ledger or CumulativeLedger()
        self._hierarchy = hierarchy or AuthorityHierarchyResolver()
        self._surfaces = surfaces or SurfaceClassifier()
        self._automation = automation or AutomationLevelPolicy()
        self._tier2: Tier2Advisor = tier2 or NullTier2Advisor()
        self._audit = audit or NullAuditLedger()
        self._basic_sanity = basic_sanity
        self._gates = ThreeGatesEvaluator()
        self._agent_id = get_settings().agent.id
        self._logger = get_logger("gatekeeper")

    @logged("gatekeeper")
    async def evaluate(self, call: ToolCall, authority: AuthorityContext) -> AuthorityDecision:
        """Evaluate the exact call against every active deterministic gate."""
        context = self._hierarchy.resolve(authority)
        amount = _amount_of(call)
        checks: list[CheckReport] = []
        three_gates: GateEvaluation | None = None
        automation: AutomationAssessment | None = None
        cumulative_total = 0.0
        threshold = 0.0
        failure: tuple[str, str, str, str] | None = None

        for check_name in self._CHECK_ORDER:
            if failure is not None:
                checks.append(self._skipped(check_name))
                continue

            if check_name == "offered":
                check, fail = self._eval_offered(call, context)
            elif check_name == "surface":
                check, fail = self._eval_surface(call, context)
            elif check_name == "actor":
                check, fail = self._eval_actor(call, context)
            elif check_name == "authority":
                check, fail = self._eval_authority(call, context)
            elif check_name == "basic_sanity":
                check, fail = self._eval_basic_sanity(call, context)
            elif check_name == "cumulative":
                check, fail, cumulative_total, threshold = self._eval_cumulative(
                    call, context, amount
                )
            elif check_name == "three_gates":
                check, fail, three_gates = self._eval_three_gates(call)
            else:
                check, fail, automation = self._eval_automation(call, amount)

            checks.extend(check)
            if fail is not None:
                failure = fail

        if failure is None:
            decision = AuthorityDecision(
                verdict="allow",
                action=call.action,
                actor=context.actor,
                reason="all deterministic execution gates passed",
                reason_code="GATEKEEPER_ALLOW",
                question="complete",
                surface=self._surfaces.classify(call.action),
                blast_radius=call.blast_radius,
                automation_level=automation.level if automation else self._automation.level_for(call.action),
                cumulative_total=cumulative_total,
                threshold=threshold,
                authority_source=context.granted_by,
                authority_rank=context.granted_by_rank,
                required_authority_rank=call.requires_authority_rank,
                max_granting_rank=self._hierarchy.max_granting_rank,
                checks=checks,
                three_gates=three_gates,
                automation=automation,
            )
        else:
            question, verdict, reason_code, reason = failure
            decision = AuthorityDecision(
                verdict=verdict,  # type: ignore[arg-type]
                action=call.action,
                actor=context.actor,
                reason=reason,
                reason_code=reason_code,
                question=question,
                surface=self._surfaces.classify(call.action),
                blast_radius=call.blast_radius,
                automation_level=automation.level if automation else self._automation.level_for(call.action),
                cumulative_total=cumulative_total,
                threshold=threshold,
                authority_source=context.granted_by,
                authority_rank=context.granted_by_rank,
                required_authority_rank=call.requires_authority_rank,
                max_granting_rank=self._hierarchy.max_granting_rank,
                checks=checks,
                three_gates=three_gates,
                automation=automation,
            )

        advisory = await self._advise(call)
        decision = decision.model_copy(update={"advisory": advisory})
        self._record(call, context, decision)
        self._logger.info(
            "Gatekeeper decision",
            agent_id=self._agent_id,
            action=call.action,
            actor=context.actor,
            verdict=decision.verdict,
            reason_code=decision.reason_code,
            question=decision.question,
            reason=decision.reason,
            checks=[_compact_check(check) for check in decision.checks],
            advisory=(
                {
                    "flagged": decision.advisory.flagged,
                    "reason": decision.advisory.reason,
                    "confidence": decision.advisory.confidence,
                }
                if decision.advisory
                else None
            ),
        )
        return decision

    def commit(
        self, call: ToolCall, authority: AuthorityContext, amount: float | None = None
    ) -> None:
        """Record a permitted action's actual consumption against the ledger."""
        spent = amount if amount is not None else _amount_of(call)
        if spent <= 0:
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

    def _eval_offered(
        self, call: ToolCall, context: AuthorityContext
    ) -> tuple[list[CheckReport], tuple[str, str, str, str] | None]:
        offered = context.offered_actions
        passed = offered is None or call.action in offered
        check = CheckReport(
            check_id="gatekeeper.offered_action",
            name="Offered action",
            status="PASS" if passed else "FAIL",
            reason_code="ACTION_OFFERED" if passed else "ACTION_NOT_OFFERED",
            reason=(
                "The exact action was in the tool set offered for this request."
                if passed
                else f"'{call.action}' was not offered for this request."
            ),
            observed={"action": call.action},
            evidence={"offered_actions": sorted(offered or [])},
        )
        fail = None if passed else (
            "offered", "deny", "ACTION_NOT_OFFERED", check.reason
        )
        return [check], fail

    def _eval_surface(
        self, call: ToolCall, context: AuthorityContext
    ) -> tuple[list[CheckReport], tuple[str, str, str, str] | None]:
        surface = self._surfaces.classify(call.action)
        writable, reason = self._surfaces.is_writable_by_agent(surface)
        status = "PASS" if writable else "ESCALATE" if surface == "human_controlled" else "FAIL"
        reason_code = (
            "SURFACE_WRITABLE"
            if writable
            else "HUMAN_CONTROLLED_SURFACE"
            if surface == "human_controlled"
            else "LOCKED_SURFACE"
        )
        check = CheckReport(
            check_id="gatekeeper.surface",
            name="Surface",
            status=status,
            reason_code=reason_code,
            reason=reason,
            observed={
                "surface": surface,
                "writable_by_agent": writable,
                "blast_radius": call.blast_radius,
            },
        )
        fail = None if writable else (
            "surface",
            "escalate" if surface == "human_controlled" else "deny",
            reason_code,
            reason,
        )
        return [check], fail

    def _eval_actor(
        self, call: ToolCall, context: AuthorityContext
    ) -> tuple[list[CheckReport], tuple[str, str, str, str] | None]:
        if self._authorizer is None:
            return [
                CheckReport(
                    check_id="gatekeeper.actor_authorization",
                    name="Actor authorization",
                    status="SKIPPED",
                    reason_code="AUTHORIZER_NOT_CONFIGURED",
                    reason="No RBAC authorizer was injected; no actor authorization decision was run.",
                )
            ], None
        outcome = self._authorizer.authorize(
            Principal(actor=context.actor, roles=context.roles), call.action
        )
        check = CheckReport(
            check_id="gatekeeper.actor_authorization",
            name="Actor authorization",
            status="PASS" if outcome.allowed else "FAIL",
            reason_code="ACTOR_AUTHORIZED" if outcome.allowed else "ACTOR_NOT_AUTHORIZED",
            reason=outcome.reason,
            observed={
                "actor": context.actor,
                "roles": context.roles,
                "action": call.action,
                "permission": outcome.permission,
                "via_elevation": outcome.via_elevation,
            },
            evidence={"authorization_allowed": outcome.allowed},
        )
        fail = None if outcome.allowed else (
            "actor", "deny", "ACTOR_NOT_AUTHORIZED", outcome.reason
        )
        return [check], fail

    def _eval_authority(
        self, call: ToolCall, context: AuthorityContext
    ) -> tuple[list[CheckReport], tuple[str, str, str, str] | None]:
        may_grant, hierarchy_reason = self._hierarchy.may_authorize(context)
        required_rank = call.requires_authority_rank
        rank_sufficient = context.granted_by_rank <= required_rank
        passed = may_grant and rank_sufficient
        reason_code = (
            "AUTHORITY_GRANTED"
            if passed
            else "AUTHORITY_SOURCE_CANNOT_GRANT"
            if not may_grant
            else "AUTHORITY_RANK_INSUFFICIENT"
        )
        reason = (
            f"'{context.granted_by}' is rank {context.granted_by_rank} and may authorize "
            f"a tool requiring rank {required_rank}."
            if passed
            else hierarchy_reason
            if not may_grant
            else (
                f"'{context.granted_by}' has rank {context.granted_by_rank}; "
                f"the tool requires rank {required_rank} or stronger authority."
            )
        )
        check = CheckReport(
            check_id="gatekeeper.authority_hierarchy",
            name="Authority hierarchy",
            status="PASS" if passed else "FAIL",
            reason_code=reason_code,
            reason=reason,
            observed={
                "granted_by": context.granted_by,
                "authority_rank": context.granted_by_rank,
                "required_rank": required_rank,
                "max_granting_rank": self._hierarchy.max_granting_rank,
            },
            evidence={"hierarchy": self._hierarchy.hierarchy},
        )
        fail = None if passed else (
            "authority", "deny", reason_code, reason
        )
        return [check], fail

    def _eval_basic_sanity(
        self, call: ToolCall, context: AuthorityContext
    ) -> tuple[list[CheckReport], tuple[str, str, str, str] | None]:
        if self._basic_sanity is None:
            return [
                CheckReport(
                    check_id="gatekeeper.basic_sanity",
                    name="Basic sanity",
                    status="SKIPPED",
                    reason_code="CHECKER_NOT_CONFIGURED",
                    reason="No agent-owned BasicSanityChecker was injected.",
                )
            ], None

        sanity_metadata = context.sanity_context.get("metadata", {})
        if not isinstance(sanity_metadata, dict):
            sanity_metadata = {}
        request = BasicSanityRequest(
            query=str(context.sanity_context.get("query", "")),
            agent_id=self._agent_id,
            session_id=context.session_id,
            actor=context.actor,
            customer_id=context.sanity_context.get("customer_id"),
            metadata={
                **sanity_metadata,
                "tool_action": call.action,
                "tool_params": dict(call.params),
            },
        )
        result = self._basic_sanity.check(request)
        reports: list[CheckReport] = []
        for finding in result.findings:
            reports.append(
                CheckReport(
                    check_id=f"gatekeeper.basic_sanity.{finding.check}",
                    name=f"Basic sanity · {finding.check.replace('_', ' ')}",
                    status="PASS" if finding.passed else "FAIL",
                    reason_code=(
                        f"BASIC_SANITY_{finding.check.upper()}_PASS"
                        if finding.passed
                        else result.reason_code
                    ),
                    reason=finding.reason,
                    observed={"check": finding.check, "passed": finding.passed},
                    evidence=finding.evidence,
                )
            )
        if not reports:
            reports.append(
                CheckReport(
                    check_id="gatekeeper.basic_sanity",
                    name="Basic sanity",
                    status="PASS" if result.passed else "FAIL",
                    reason_code=result.reason_code,
                    reason=result.message or result.reason_code,
                    observed={"passed": result.passed},
                    evidence=result.facts,
                )
            )
        fail = None
        if not result.passed:
            failed = next((item for item in reports if item.status == "FAIL"), reports[-1])
            fail = ("basic_sanity", "deny", result.reason_code, result.message or failed.reason)
        return reports, fail

    def _eval_cumulative(
        self, call: ToolCall, context: AuthorityContext, amount: float
    ) -> tuple[list[CheckReport], tuple[str, str, str, str] | None, float, float]:
        if not self._ledger.enabled:
            return [
                CheckReport(
                    check_id="gatekeeper.cumulative_limit",
                    name="Cumulative limit",
                    status="SKIPPED",
                    reason_code="CUMULATIVE_POLICY_DISABLED",
                    reason="Cumulative spend accounting is disabled by configuration.",
                )
            ], None, 0.0, 0.0

        action_class = self._ledger.class_of(call.action)
        ledger_key = _ledger_key(context)
        previous = self._ledger.window_total(ledger_key, action_class)
        state = self._ledger.state(ledger_key, action_class, pending=amount)
        projected = state.window_total
        threshold = state.threshold
        passed = not state.exceeded
        check = CheckReport(
            check_id="gatekeeper.cumulative_limit",
            name="Cumulative limit",
            status="PASS" if passed else "FAIL",
            reason_code="CUMULATIVE_WITHIN_LIMIT" if passed else "CUMULATIVE_LIMIT_EXCEEDED",
            reason=(
                f"Projected {action_class} spend is ${projected:,.2f} within the "
                f"${threshold:,.2f} rolling limit."
                if passed
                else f"Projected {action_class} spend of ${projected:,.2f} exceeds the "
                f"${threshold:,.2f} rolling limit."
            ),
            observed={
                "action_class": action_class,
                "ledger_key": ledger_key,
                "previous": previous,
                "proposed": amount,
                "projected": projected,
                "threshold": threshold,
                "remaining_before": max(0.0, threshold - previous),
                "remaining_after": max(0.0, threshold - projected),
                "entry_count": state.entry_count,
            },
        )
        fail = None if passed else (
            "cumulative", "deny", "CUMULATIVE_LIMIT_EXCEEDED", check.reason
        )
        return [check], fail, projected, threshold

    def _eval_three_gates(
        self, call: ToolCall
    ) -> tuple[list[CheckReport], tuple[str, str, str, str] | None, GateEvaluation]:
        evaluation = self._gates.evaluate(
            path_cardinality=call.path_cardinality,
            blast_radius=call.blast_radius,
            certifiable=call.certifiable,
        )
        reports = [
            evaluation.path_check
            or self._fallback_gate_check("path_cardinality", evaluation.path_cardinality),
            evaluation.blast_radius_check
            or self._fallback_gate_check("blast_radius", evaluation.blast_radius),
            evaluation.certifiability_check
            or self._fallback_gate_check("certifiability", evaluation.certifiable),
        ]
        fail = None if evaluation.passes else (
            "three_gates", "deny", "THREE_GATES_FAILED", evaluation.reason
        )
        return reports, fail, evaluation

    def _eval_automation(
        self, call: ToolCall, amount: float
    ) -> tuple[list[CheckReport], tuple[str, str, str, str] | None, AutomationAssessment]:
        assessment = self._automation.assess(call.action, amount)
        passed = not assessment.requires_human
        status = "PASS" if passed else "ESCALATE"
        check = CheckReport(
            check_id="gatekeeper.automation_policy",
            name="Automation level",
            status=status,
            reason_code="AUTOMATION_ALLOWED" if passed else "HUMAN_REQUIRED",
            reason=(
                f"Sheridan level {assessment.level}; "
                + ("human approval required" if assessment.requires_human else "autonomous execution permitted")
            ),
            observed={
                "sheridan_level": assessment.level,
                "level_meaning": assessment.level_meaning,
                "amount": assessment.amount,
                "auto_approval_ceiling": assessment.auto_approval_ceiling,
                "amount_within_ceiling": assessment.amount_within_ceiling,
                "automation_floor": assessment.automation_floor,
                "level_meets_floor": assessment.level_meets_floor,
                "human_required": assessment.requires_human,
            },
        )
        fail = None if passed else (
            "automation", "escalate", "HUMAN_REQUIRED", assessment.reason
        )
        return [check], fail, assessment

    @staticmethod
    def _fallback_gate_check(name: str, value: object) -> CheckReport:
        return CheckReport(
            check_id=f"gatekeeper.three_gates.{name}",
            name=name.replace("_", " ").title(),
            status="PASS",
            reason_code="THREE_GATES_ACCEPTED",
            reason="Dimension accepted by the aggregate evaluator.",
            observed={"value": value},
        )

    def _skipped(self, check_name: str) -> CheckReport:
        display = check_name.replace("_", " ").title()
        return CheckReport(
            check_id=f"gatekeeper.{check_name}",
            name=display,
            status="SKIPPED",
            reason_code="SHORT_CIRCUITED",
            reason="Not evaluated because an earlier Gatekeeper check already blocked or escalated the call.",
        )

    async def _advise(self, call: ToolCall) -> AdvisoryFlag:
        """Attach the Tier-2 note. It annotates; it does not decide."""
        return await self._tier2.flag(call)

    def _record(
        self, call: ToolCall, context: AuthorityContext, decision: AuthorityDecision
    ) -> None:
        """Write the decision to the audit trail — every decision, always."""
        self._audit.record(
            AuditEntry(
                trace_id=get_current_trace_id() or "",
                actor=context.actor,
                action=call.action,
                authority_context={
                    **context.model_dump(mode="json"),
                    "agent_id": self._agent_id,
                },
                model_version=get_settings().llm.providers[get_settings().llm.provider].model,
                decision=decision.verdict,
                rationale=decision.reason,
            )
        )


def _amount_of(call: ToolCall) -> float:
    """The monetary size of a call, if it has one."""
    for name in _AMOUNT_PARAMS:
        value = call.params.get(name)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def _ledger_key(context: AuthorityContext) -> str:
    """Which cumulative budget this call consumes from."""
    return context.cumulative_ledger_key or context.actor
