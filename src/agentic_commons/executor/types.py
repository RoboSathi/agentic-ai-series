"""Contracts for the deterministic execution boundary."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from agentic_commons.gatekeeper.types import AuthorityContext, AuthorityDecision
from agentic_commons.observability.types import utc_now
from agentic_commons.tools.types import ToolCall, ToolResult

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
