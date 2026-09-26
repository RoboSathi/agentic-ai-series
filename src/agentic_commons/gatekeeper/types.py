"""Gatekeeper value types (LLD §11).

The type distinction that carries the whole Tier-1/Tier-2 rule:
`AuthorityDecision` can deny; `AdvisoryFlag` cannot.  A Tier-2 advisor's
signature returns the latter, so a semantic classifier is structurally
incapable of blocking an action no matter what it concludes.

This module also carries the canonical, human-readable decision report used
by the TUI and the file logs.  The Gatekeeper evaluates deterministic checks;
the report records exactly which checks ran, which passed, which failed, and
which were skipped because of short-circuiting.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from agentic_commons.gatekeeper.mandate import MandateSet
from agentic_commons.money import Money
from agentic_commons.observability.types import utc_now

Verdict = Literal["allow", "deny", "escalate"]
CheckStatus = Literal["PASS", "FAIL", "ESCALATE", "SKIPPED"]


class AuthorityContext(BaseModel):
    """Trusted request authority plus a small opaque context for agent rules."""

    actor: str
    granted_by: str = "user_task"
    cumulative_ledger_key: str = ""
    session_id: str = ""
    offered_actions: list[str] | None = None
    request_context: dict[str, Any] = Field(default_factory=dict)
    # What the person actually asked for this turn. `None` means no
    # issuer is wired, which is not the same as an empty set: an empty
    # set says "they asked for nothing", and a write is then refused.
    mandates: "MandateSet | None" = None
    turn: int = 0


class AdvisoryFlag(BaseModel):
    """A Tier-2 observation. Annotates and escalates; never blocks."""

    flagged: bool = False
    reason: str = ""
    confidence: float = 0.0
    advisory_only: bool = True


class CheckReport(BaseModel):
    """One deterministic check in a decision report."""

    check_id: str
    name: str
    status: CheckStatus
    reason_code: str = ""
    reason: str = ""
    observed: dict[str, Any] = Field(default_factory=dict)
    evidence: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=utc_now)
    duration_ms: float | None = None


class AuthorityDecision(BaseModel):
    """The Gatekeeper's ruling on one tool call."""

    verdict: Verdict
    action: str
    actor: str
    reason: str
    reason_code: str = ""
    question: str = ""
    # Where the actor stood against its rolling budget, when one applied.
    cumulative_total: Money | None = None
    threshold: Money | None = None
    authority_source: str = ""
    authority_rank: int | None = None
    required_authority_rank: int | None = None
    max_granting_rank: int | None = None
    checks: list[CheckReport] = Field(default_factory=list)
    advisory: AdvisoryFlag | None = None
    decided_at: datetime = Field(default_factory=utc_now)

    @property
    def allowed(self) -> bool:
        return self.verdict == "allow"


class LedgerEntry(BaseModel):
    """One recorded consumption against a cumulative budget."""

    agent_id: str = ""
    ledger_key: str
    action_class: str
    actor: str
    amount: Money
    trace_id: str = ""
    recorded_at: datetime = Field(default_factory=utc_now)


class LedgerState(BaseModel):
    """Where an actor stands against a threshold inside the window."""

    ledger_key: str
    action_class: str
    window_total: Money
    # None: this class has no finite threshold (see `threshold_configured`).
    threshold: Money | None
    entry_count: int
    exceeded: bool

    @property
    def remaining(self) -> Money | None:
        """Budget left before the threshold trips (None when unbounded)."""
        if self.threshold is None:
            return None
        return self.threshold - self.window_total if self.threshold > self.window_total else Money.zero()


class AuthorityContextMeta(BaseModel):
    """Serialized authority context for the audit trail."""

    data: dict[str, Any] = Field(default_factory=dict)


AuthorityContext.model_rebuild()
