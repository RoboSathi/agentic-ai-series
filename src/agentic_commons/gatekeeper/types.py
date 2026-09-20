"""Gatekeeper value types (LLD §11).

The type distinction that carries the whole Tier-1/Tier-2 rule:
`AuthorityDecision` can deny; `AdvisoryFlag` cannot. A Tier-2 advisor's
signature returns the latter, so a semantic classifier is structurally
incapable of blocking an action no matter what it concludes — enforcement
by the type system rather than by reviewer discipline.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from agentic_commons.observability.types import utc_now
from agentic_commons.tools.types import BlastRadius, SurfaceClass

Verdict = Literal["allow", "deny", "escalate"]


class AuthorityContext(BaseModel):
    """Who is acting, under whose authority, and against which ledger.

    `granted_by_rank` is the rank of the source that granted this
    authority — not the actor's own standing. A user_task (rank 5) can
    authorize within its scope; a tool_observation (rank 7) cannot
    authorize anything, which is what makes injected instructions inert.
    """

    actor: str
    granted_by: str = "user_task"
    granted_by_rank: int = 5
    cumulative_ledger_key: str = ""
    roles: list[str] = Field(default_factory=list)
    session_id: str = ""
    # Actions this turn was offered. Narrowing the toolset is an authority
    # decision made before the model saw the turn, so it belongs here and
    # is checked — and audited — like any other.
    #
    # None and [] are different, and conflating them is a fail-open bug:
    # None means no narrowing was applied, while an empty list means the
    # turn was deliberately offered nothing — which is precisely the case
    # for a conversational message, and exactly when a refund must not
    # slip through.
    offered_actions: list[str] | None = None


class AdvisoryFlag(BaseModel):
    """A Tier-2 observation. Annotates and escalates; never blocks."""

    flagged: bool = False
    reason: str = ""
    confidence: float = 0.0
    advisory_only: bool = True


class AuthorityDecision(BaseModel):
    """The Gatekeeper's ruling on one tool call."""

    verdict: Verdict
    action: str
    actor: str
    reason: str
    question: str = ""
    surface: SurfaceClass = "editable"
    blast_radius: BlastRadius = "single_record"
    automation_level: int = 0
    cumulative_total: float = 0.0
    threshold: float = 0.0
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
    amount: float
    trace_id: str = ""
    recorded_at: datetime = Field(default_factory=utc_now)


class LedgerState(BaseModel):
    """Where an actor stands against a threshold inside the window."""

    ledger_key: str
    action_class: str
    window_total: float
    threshold: float
    entry_count: int
    exceeded: bool

    @property
    def remaining(self) -> float:
        """Budget left before the threshold trips."""
        return max(0.0, self.threshold - self.window_total)


class AutomationAssessment(BaseModel):
    """Sheridan level for an action, and whether a human is required (E03)."""

    action: str
    level: int
    requires_human: bool
    reason: str = ""
    amount: float = 0.0


class GateEvaluation(BaseModel):
    """The Three Gates: path cardinality × blast radius × certifiability (E15)."""

    path_cardinality: Literal["single", "bounded", "open"] = "bounded"
    blast_radius: BlastRadius = "single_record"
    certifiable: bool = True
    passes: bool = True
    reason: str = ""


class AuthorityContextMeta(BaseModel):
    """Serialized authority context for the audit trail."""

    data: dict[str, Any] = Field(default_factory=dict)
