"""Scoping vocabulary: archetypes, the coverage checklist, the Scope Brief.

The Scope Brief is the *Elephant*: the frozen memory of what the user
actually asked for. Everything downstream — the Rubric, the plan, the
audit — is derived from it and checked against it, so it is typed and
immutable rather than a paragraph of chat history.
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from research_agent.domain.budget import Budget

Archetype = Literal[
    "decision_support", "landscape_brief", "fact_verification", "explainer", "general"
]
ItemStatus = Literal["open", "covered", "waived"]
# How big a run and its report are: each size is a profile in config (`agent.depths`).
Depth = Literal["small", "medium", "large"]
DEFAULT_DEPTH: Depth = "small"
KindDecision = Literal["scopable", "refuse"]
# What decided a research kind: a pattern, the encoder, or the model.
KindSource = Literal["rule", "encoder", "llm"]
UnknownReason = Literal["waived", "closed_early", "turn_cap"]

# The kinds a research run can take; `general` when no specialised one fits.
RESEARCH_ARCHETYPES: tuple[Archetype, ...] = (
    "decision_support",
    "landscape_brief",
    "fact_verification",
    "explainer",
    "general",
)
GENERAL: Archetype = "general"


class Choice(BaseModel):
    """One answer offered from a menu: what the user sees, and what is recorded."""

    model_config = ConfigDict(frozen=True)

    label: str
    value: str


class CoverageItem(BaseModel):
    """One thing the interview must establish (or the user must waive)."""

    item_id: str
    # What the interview needs to learn, as a short noun phrase; the interviewer
    # words the actual question for this request, so this is never shown verbatim.
    goal: str
    # A closed set of answers, offered as a menu beside free text (empty: open-ended).
    choices: tuple[Choice, ...] = ()
    required: bool = True
    # An optional item the interviewer may drop as irrelevant to a request; False means it is
    # always asked (a menu makes that one click), e.g. how big the research should be.
    skippable: bool = True
    status: ItemStatus = "open"
    value: str = ""
    # A private item's value never leaves the process (queries, URLs).
    private: bool = False
    # Which Brief field the answer fills, e.g. `criteria`.
    brief_field: str = ""
    # Whether the answer is a list (comma/semicolon separated).
    multi: bool = False


class KindVerdict(BaseModel):
    """The research kind a `/research` question was read as (or a refusal)."""

    decision: KindDecision
    archetype: Archetype | None = None
    topic: str = ""
    reason: str = ""
    source: KindSource = "llm"
    # The model that decided, when one did (encoder checkpoint or LLM).
    source_model: str = ""
    # A time window stated in the message ("in the last 1 month" -> 30).
    recent_days: int | None = None
    # What the intent classifier made of the message, when it was asked, whether or not it was
    # the one that decided: its top label, its share of the vote and the runner-up's.
    encoder_label: str = ""
    encoder_share: float = 0.0
    encoder_runner_up: float = 0.0


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
    # The span of events the research covers, resolved in code from a window the user stated
    # ("the last 20 years" on 2026-10-04 → 2006-10-04 … 2026-10-04): never left to a model,
    # which would count back from its training-era "now".
    period_start: date | None = None
    period_end: date | None = None
    geography: str = ""
    depth: Depth = DEFAULT_DEPTH
    deliverable_format: str = ""
    budget: Budget
    known_unknowns: tuple[KnownUnknown, ...] = ()
    # Strings the user marked private; never allowed into an outbound query or URL.
    private_terms: tuple[str, ...] = ()

    @property
    def subjects(self) -> tuple[str, ...]:
        """What this archetype researches item by item: options, claims or concepts."""
        return self.options or self.claims or self.concepts
