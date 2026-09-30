"""Scoping vocabulary: archetypes, the coverage checklist, the Scope Brief.

The Scope Brief is the *Elephant*: the frozen memory of what the user
actually asked for. Everything downstream — the Rubric, the plan, the
audit — is derived from it and checked against it, so it is typed and
immutable rather than a paragraph of chat history.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from research_agent.domain.budget import Budget

Archetype = Literal[
    "decision_support", "landscape_brief", "fact_verification", "explainer", "general", "qa"
]
# What the user chooses between: a quick cited answer, or a scoped research run.
Mode = Literal["qa", "research"]
ItemStatus = Literal["open", "covered", "waived"]
Depth = Literal["overview", "standard", "deep"]
GateDecision = Literal["scopable", "needs_reframe", "refuse"]
UnknownReason = Literal["waived", "closed_early", "turn_cap"]

# The kinds a Research run can take; `general` when no specialised one fits.
RESEARCH_ARCHETYPES: tuple[Archetype, ...] = (
    "decision_support",
    "landscape_brief",
    "fact_verification",
    "explainer",
    "general",
)
# Every kind, including `qa` — the Q&A mode's fast path (no interview).
ARCHETYPES: tuple[Archetype, ...] = (*RESEARCH_ARCHETYPES, "qa")
QA: Archetype = "qa"
GENERAL: Archetype = "general"


class CoverageItem(BaseModel):
    """One thing the interview must establish (or the user must waive)."""

    item_id: str
    prompt: str
    required: bool = True
    status: ItemStatus = "open"
    value: str = ""
    # A private item's value never leaves the process (queries, URLs).
    private: bool = False
    # Which Brief field the answer fills, e.g. `criteria`.
    brief_field: str = ""
    # Whether the answer is a list (comma/semicolon separated).
    multi: bool = False


class GateVerdict(BaseModel):
    """The scopability gate's decision on a first message."""

    decision: GateDecision
    mode: Mode = "research"
    archetype: Archetype | None = None
    topic: str = ""
    reason: str = ""
    # A time window stated in the message ("in the last 1 month" -> 30).
    recent_days: int | None = None


class SourcePolicy(BaseModel):
    """Which sources count, and how many independent ones are needed."""

    model_config = ConfigDict(frozen=True)

    preferred_domains: tuple[str, ...] = ()
    blocked_domains: tuple[str, ...] = ()
    primary_sources_preferred: bool = False
    min_distinct_domains: int = 1


class KnownUnknown(BaseModel):
    """A scope item the user chose not to settle, and why it stayed open."""

    model_config = ConfigDict(frozen=True)

    item_id: str
    question: str
    reason: UnknownReason


class ScopeBrief(BaseModel):
    """The frozen, typed output of scoping."""

    model_config = ConfigDict(frozen=True)

    archetype: Archetype
    topic: str
    purpose: str = ""
    decision: str = ""
    criteria: tuple[str, ...] = ()
    options: tuple[str, ...] = ()
    claims: tuple[str, ...] = ()
    concepts: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    source_policy: SourcePolicy = Field(default_factory=SourcePolicy)
    recency_days: int | None = None
    geography: str = ""
    depth: Depth = "standard"
    deliverable_format: str = ""
    budget: Budget
    known_unknowns: tuple[KnownUnknown, ...] = ()
    # Strings the user marked private; never allowed into an outbound query or URL.
    private_terms: tuple[str, ...] = ()

    @property
    def subjects(self) -> tuple[str, ...]:
        """What this archetype researches item by item: options, claims or concepts."""
        return self.options or self.claims or self.concepts
