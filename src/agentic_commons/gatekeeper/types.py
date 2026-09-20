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

from agentic_commons.observability.types import utc_now
from agentic_commons.tools.types import BlastRadius, SurfaceClass

Verdict = Literal["allow", "deny", "escalate"]
CheckStatus = Literal["PASS", "FAIL", "ESCALATE", "SKIPPED"]


class AuthorityContext(BaseModel):
    """Who is acting, under whose authority, and against which ledger.

    `granted_by_rank` is always resolved from `granted_by`; a caller-supplied
    rank is never trusted.  `sanity_context` is deliberately generic so the
    common Gatekeeper can pass request-scoped facts to an agent-owned
    ``BasicSanityChecker`` without importing the agent's domain model.
    """

    actor: str
    granted_by: str = "user_task"
    granted_by_rank: int = 5
    cumulative_ledger_key: str = ""
    roles: list[str] = Field(default_factory=list)
    session_id: str = ""
    offered_actions: list[str] | None = None
    sanity_context: dict[str, Any] = Field(default_factory=dict)


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
    surface: SurfaceClass = "editable"
    blast_radius: BlastRadius = "single_record"
    automation_level: int = 0
    cumulative_total: float = 0.0
    threshold: float = 0.0
    authority_source: str = ""
    authority_rank: int | None = None
    required_authority_rank: int | None = None
    max_granting_rank: int | None = None
    checks: list[CheckReport] = Field(default_factory=list)
    three_gates: "GateEvaluation | None" = None
    automation: "AutomationAssessment | None" = None
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
    policy_enabled: bool = True
    automation_floor: int = 5
    level_meets_floor: bool = True
    auto_approval_ceiling: float = float("inf")
    amount_within_ceiling: bool = True
    level_meaning: str = ""


class GateEvaluation(BaseModel):
    """Path cardinality × blast radius × certifiability (E15)."""

    path_cardinality: Literal["single", "bounded", "open"] = "bounded"
    blast_radius: BlastRadius = "single_record"
    certifiable: bool = True
    passes: bool = True
    reason: str = ""
    path_check: CheckReport | None = None
    blast_radius_check: CheckReport | None = None
    certifiability_check: CheckReport | None = None


class AuthorityContextMeta(BaseModel):
    """Serialized authority context for the audit trail."""

    data: dict[str, Any] = Field(default_factory=dict)
