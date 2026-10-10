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

from pydantic import BaseModel, Field, model_validator

from agent_platform.gatekeeper.mandate import MandateSet
from agent_platform.observability.types import utc_now
from agent_platform.primitives.money import Money

Verdict = Literal["allow", "deny", "escalate"]
CheckStatus = Literal["PASS", "FAIL", "ESCALATE", "SKIPPED"]
# Where an action stands in agency terms: what the principal may do about it.
#   VOID        refused, and nobody can adopt it: the agent had no authority at all.
#   RATIFIABLE  beyond the agent's own bounds; a person with authority may adopt it.
#   RATIFIED    a person adopted a RATIFIABLE action, and it ran on their authority.
#   VOIDABLE    within authority and executed, but undoable until its window closes.
#   VALID       within authority and executed, for good.
#   UNGOVERNED  executed with the Gatekeeper withheld: nobody ruled on it.
Standing = Literal["VOID", "RATIFIABLE", "RATIFIED", "VOIDABLE", "VALID", "UNGOVERNED"]
# The standing a verdict implies when nothing more is known. An allow is
# VALID unless the Gatekeeper knows the action is voidable.
_STANDING_OF_VERDICT: dict[str, Standing] = {
    "allow": "VALID",
    "deny": "VOID",
    "escalate": "RATIFIABLE",
}


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
    checks: list[CheckReport] = Field(default_factory=list)
    advisory: AdvisoryFlag | None = None
    decided_at: datetime = Field(default_factory=utc_now)
    # Derived from the verdict when not given (see `_STANDING_OF_VERDICT`).
    standing: Standing | None = None

    @model_validator(mode="after")
    def _derive_standing(self) -> "AuthorityDecision":
        if self.standing is None:
            self.standing = _STANDING_OF_VERDICT[self.verdict]
        return self

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
        return (
            self.threshold - self.window_total
            if self.threshold > self.window_total
            else Money.zero()
        )


AuthorityContext.model_rebuild()
