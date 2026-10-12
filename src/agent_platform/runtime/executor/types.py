"""Contracts for the deterministic execution boundary."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from agent_platform.gatekeeper.types import AuthorityContext, AuthorityDecision
from agent_platform.observability.types import utc_now
from agent_platform.tools.types import ToolCall, ToolResult

# The Gatekeeper verdict a referral is the harness's answer to.
_ESCALATE = "escalate"

ExecutionStatus = Literal["completed", "failed", "timed_out", "replayed"]
ExecutionPhase = Literal["started", "finished"]


class ApprovedToolCall(BaseModel):
    """Opaque hand-off from Gatekeeper to Executor."""

    call: ToolCall
    decision: AuthorityDecision
    authority: AuthorityContext

    @classmethod
    def from_decision(
        cls,
        call: ToolCall,
        decision: AuthorityDecision,
        authority: AuthorityContext,
    ) -> "ApprovedToolCall":
        """Construct only from an allow decision for the exact same action."""
        if not decision.allowed:
            raise ValueError("Cannot create ApprovedToolCall from a non-allow decision")
        if decision.action != call.action:
            raise ValueError(
                f"Gatekeeper action '{decision.action}' does not match '{call.action}'"
            )
        return cls(call=call, decision=decision, authority=authority)

    @classmethod
    def ungoverned(cls, call: ToolCall, authority: AuthorityContext) -> "ApprovedToolCall":
        """An execution with no Gatekeeper standing behind it.

        Only reachable when a deployment has explicitly withheld the
        `gatekeeper` feature, which episodes do to show what the control
        was holding back. The synthesized decision says exactly that
        rather than mimicking an approval — whatever reads this later
        (the audit trail especially) must be able to tell an execution
        nobody authorized from one somebody did.
        """
        return cls(
            call=call,
            decision=AuthorityDecision(
                verdict="allow",
                action=call.action,
                actor=authority.actor,
                reason="Executed without authorization: the gatekeeper feature is switched off.",
                reason_code="GATEKEEPER_DISABLED",
                question="ungoverned",
                standing="UNGOVERNED",
            ),
            authority=authority,
        )

    @classmethod
    def referral(
        cls,
        original: ToolCall,
        referral: ToolCall,
        decision: AuthorityDecision,
        authority: AuthorityContext,
    ) -> "ApprovedToolCall":
        """The harness's own move after an escalation: send the call to a person.

        Built only from an *escalate* decision for `original`, and it
        approves `referral` — the queueing write — never the escalated
        action itself. The decision says so, so the audit trail can tell a
        referral from an approval, the way it tells an ungoverned call.
        Running it through the Executor gives the queue write the same
        idempotency, deadline and trace as every other side effect.
        """
        if decision.verdict != _ESCALATE:
            raise ValueError("A referral is built only from an escalate decision")
        if decision.action != original.action:
            raise ValueError(
                f"Gatekeeper action '{decision.action}' does not match '{original.action}'"
            )
        return cls(
            call=referral,
            decision=AuthorityDecision(
                verdict="allow",
                action=referral.action,
                actor=authority.actor,
                reason=(
                    f"Referral of the escalated '{original.action}' for human review; "
                    "the action itself was not approved."
                ),
                reason_code="ESCALATION_REFERRAL",
                question="referral",
            ),
            authority=authority,
        )


class ExecutionOutcome(BaseModel):
    """Normalized result plus execution metadata."""

    result: ToolResult
    status: ExecutionStatus
    replayed: bool = False
    timed_out: bool = False
    idempotency_key: str = ""
    duration_ms: float = 0.0
    started_at: datetime = Field(default_factory=utc_now)
    ended_at: datetime = Field(default_factory=utc_now)


class ExecutionAuditEvent(BaseModel):
    """Hook payload emitted by the generic Executor."""

    action: str
    actor: str
    phase: ExecutionPhase
    status: ExecutionStatus | None = None
    idempotency_key: str = ""
    replayed: bool = False
    timed_out: bool = False
    duration_ms: float = 0.0
    result: ToolResult = Field(default_factory=ToolResult)
    error: str | None = None
    timestamp: datetime = Field(default_factory=utc_now)
