"""The Gatekeeper — the single, always-invoked mediation point (E12, LLD §11).

Four questions, in order, short-circuiting on the first denial:

    1. What surface does this touch?      → SurfaceClassifier
    2. Who is asking?                     → Authorizer (RBAC)
    3. Under whose authority?             → AuthorityHierarchyResolver
    4. Has the cumulative state crossed a threshold? → CumulativeLedger

The order is deliberate and cheapest-first, but more importantly each
question is one a per-call permission check cannot answer alone. A system
that only asks question 2 permits the injected instruction (question 3)
and the structuring attack (question 4), and both were individually
"authorized" every step of the way.

Two invariants hold whatever the configuration says:

  * The Gatekeeper is always invoked. There is no `enabled` flag here —
    an off switch on the mediation point is an execution path around it
    (PRD G1).
  * Tier 2 never decides. Its flag is attached to the decision and can
    raise an escalation, never grant or refuse one.
"""

from typing import Protocol, runtime_checkable

from agentic_commons.audit.ledger import AuditLedger, NullAuditLedger
from agentic_commons.audit.types import AuditEntry
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
    LedgerEntry,
)
from agentic_commons.logging import get_logger, logged
from agentic_commons.logging.trace_context import get_current_trace_id
from agentic_commons.security.auth.types import AccessDecision, Principal
from agentic_commons.tools.types import ToolCall

# The parameter naming a call's monetary size, for ledger and automation
# checks. Anything without it is treated as a zero-amount action.
_AMOUNT_PARAMS = ("amount", "amount_usd", "value")


@runtime_checkable
class RoleAuthorizer(Protocol):
    """Answers "may this principal perform this action?" (RBAC).

    Structurally satisfied by `security.Authorizer`, injected so the
    Gatekeeper composes the real RBAC implementation without owning it.
    """

    def authorize(self, principal: Principal, action: str) -> AccessDecision: ...


@runtime_checkable
class Gatekeeper(Protocol):
    """The mediation point every execution path passes through."""

    async def evaluate(self, call: ToolCall, authority: AuthorityContext) -> AuthorityDecision: ...


class GatekeeperImpl:
    """Reference Monitor over the four questions (Facade + Chain of Responsibility)."""

    def __init__(
        self,
        authorizer: RoleAuthorizer | None = None,
        ledger: CumulativeLedger | None = None,
        hierarchy: AuthorityHierarchyResolver | None = None,
        surfaces: SurfaceClassifier | None = None,
        automation: AutomationLevelPolicy | None = None,
        tier2: Tier2Advisor | None = None,
        audit: AuditLedger | None = None,
    ) -> None:
        self._authorizer = authorizer
        self._ledger = ledger or CumulativeLedger()
        self._hierarchy = hierarchy or AuthorityHierarchyResolver()
        self._surfaces = surfaces or SurfaceClassifier()
        self._automation = automation or AutomationLevelPolicy()
        self._tier2: Tier2Advisor = tier2 or NullTier2Advisor()
        self._audit = audit or NullAuditLedger()
        self._gates = ThreeGatesEvaluator()
        self._logger = get_logger("gatekeeper")

    @logged("gatekeeper")
    async def evaluate(self, call: ToolCall, authority: AuthorityContext) -> AuthorityDecision:
        """Answer all four questions and rule on the call."""
        context = self._hierarchy.resolve(authority)
        amount = _amount_of(call)

        decision = (
            self._check_surface(call, context)
            or self._check_actor(call, context)
            or self._check_authority(call, context)
            or self._check_cumulative(call, context, amount)
            or self._check_automation(call, context, amount)
            or self._allow(call, context, amount)
        )

        decision = decision.model_copy(update={"advisory": await self._advise(call)})
        self._record(call, context, decision)
        return decision

    def commit(
        self, call: ToolCall, authority: AuthorityContext, amount: float | None = None
    ) -> None:
        """Record a permitted action's consumption against the ledger.

        Separate from `evaluate` on purpose: the ledger records what
        actually happened, and a call that was allowed but then failed
        must not consume budget it never spent.
        """
        spent = amount if amount is not None else _amount_of(call)
        if spent <= 0:
            return
        self._ledger.record_and_check(
            LedgerEntry(
                ledger_key=_ledger_key(authority),
                action_class=self._ledger.class_of(call.action),
                actor=authority.actor,
                amount=spent,
                trace_id=get_current_trace_id() or "",
            )
        )

    def _check_surface(self, call: ToolCall, context: AuthorityContext) -> AuthorityDecision | None:
        """Question 1 — what surface does this touch?"""
        surface = self._surfaces.classify(call.action)
        writable, reason = self._surfaces.is_writable_by_agent(surface)
        if writable:
            return None
        verdict = "escalate" if surface == "human_controlled" else "deny"
        return AuthorityDecision(
            verdict=verdict,  # type: ignore[arg-type]
            action=call.action,
            actor=context.actor,
            reason=reason,
            question="surface",
            surface=surface,
            blast_radius=call.blast_radius,
        )

    def _check_actor(self, call: ToolCall, context: AuthorityContext) -> AuthorityDecision | None:
        """Question 2 — who is asking?"""
        if self._authorizer is None:
            return None
        outcome = self._authorizer.authorize(
            Principal(actor=context.actor, roles=context.roles), call.action
        )
        if outcome.allowed:
            return None
        return AuthorityDecision(
            verdict="deny",
            action=call.action,
            actor=context.actor,
            reason=outcome.reason,
            question="actor",
            surface=self._surfaces.classify(call.action),
            blast_radius=call.blast_radius,
        )

    def _check_authority(
        self, call: ToolCall, context: AuthorityContext
    ) -> AuthorityDecision | None:
        """Question 3 — under whose authority?

        This is the question that makes injected instructions inert: the
        content may be perfectly persuasive and still arrive at a rank
        that cannot authorize anything.
        """
        may_grant, reason = self._hierarchy.may_authorize(context)
        if may_grant:
            return None
        return AuthorityDecision(
            verdict="deny",
            action=call.action,
            actor=context.actor,
            reason=reason,
            question="authority",
            surface=self._surfaces.classify(call.action),
            blast_radius=call.blast_radius,
        )

    def _check_cumulative(
        self, call: ToolCall, context: AuthorityContext, amount: float
    ) -> AuthorityDecision | None:
        """Question 4 — has the cumulative state crossed a threshold?"""
        if not self._ledger.enabled:
            return None
        action_class = self._ledger.class_of(call.action)
        state = self._ledger.state(_ledger_key(context), action_class, pending=amount)
        if not state.exceeded:
            return None
        return AuthorityDecision(
            verdict="deny",
            action=call.action,
            actor=context.actor,
            reason=(
                f"cumulative {action_class} of ${state.window_total:,.2f} would exceed the "
                f"${state.threshold:,.2f} window threshold across {state.entry_count} prior actions"
            ),
            question="cumulative",
            surface=self._surfaces.classify(call.action),
            blast_radius=call.blast_radius,
            cumulative_total=state.window_total,
            threshold=state.threshold,
        )

    def _check_automation(
        self, call: ToolCall, context: AuthorityContext, amount: float
    ) -> AuthorityDecision | None:
        """Sheridan gate — permitted, but does a human still have to see it?"""
        assessment = self._automation.assess(call.action, amount)
        if not assessment.requires_human:
            return None
        return AuthorityDecision(
            verdict="escalate",
            action=call.action,
            actor=context.actor,
            reason=assessment.reason,
            question="automation",
            surface=self._surfaces.classify(call.action),
            blast_radius=call.blast_radius,
            automation_level=assessment.level,
        )

    def _allow(self, call: ToolCall, context: AuthorityContext, amount: float) -> AuthorityDecision:
        """Every question passed."""
        state = self._ledger.state(
            _ledger_key(context), self._ledger.class_of(call.action), pending=amount
        )
        return AuthorityDecision(
            verdict="allow",
            action=call.action,
            actor=context.actor,
            reason="all four questions passed",
            question="complete",
            surface=self._surfaces.classify(call.action),
            blast_radius=call.blast_radius,
            automation_level=self._automation.level_for(call.action),
            cumulative_total=state.window_total,
            threshold=state.threshold,
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
                authority_context=context.model_dump(mode="json"),
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
    """Which cumulative budget this call consumes from.

    Defaults to the actor. Keying per actor is what makes structuring
    visible — eighty calls from one agent share one running total, which
    is precisely what splitting the amount was meant to avoid.
    """
    return context.cumulative_ledger_key or context.actor
