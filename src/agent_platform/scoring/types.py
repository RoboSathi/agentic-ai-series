"""The scoring cascade's value types: a claim, and how it was resolved (LLD §18)."""

from typing import Any, Literal

from pydantic import BaseModel, Field

ScorerTier = Literal[
    "exact_assertion",
    "deterministic_rule",
    "embedding_similarity",
    "nli_entailment",
    "calibrated_llm_judge",
    "human_adjudication",
]

VerdictLabel = Literal["supported", "contradicted", "unresolved", "deferred"]


class Claim(BaseModel):
    """One assertion to be resolved against evidence.

    Deliberately generic: a claim is a retrieval hit's relevance, a tool's
    fitness for a task, a model answer's groundedness, or an eval's
    expected value. One vocabulary, one cascade.
    """

    text: str
    evidence: list[str] = Field(default_factory=list)
    expected: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Verdict(BaseModel):
    """How a claim was resolved, and by which tier.

    `tier` is the point of the whole exercise: it is what makes
    `nollm_scorer_ratio` measurable rather than aspirational.
    """

    label: VerdictLabel
    score: float
    tier: ScorerTier
    rationale: str = ""
    cost_usd: float = 0.0
