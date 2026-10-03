"""Eval value types (LLD §18)."""

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

PyramidLevel = Literal["component", "integration", "golden_task", "production_shadow"]

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
    `nollm_scorer_ratio` measurable rather than aspirational (E10).
    """

    label: VerdictLabel
    score: float
    tier: ScorerTier
    rationale: str = ""
    cost_usd: float = 0.0


class ScoreResult(BaseModel):
    """A pass/fail judgement with the score that produced it."""

    passed: bool
    score: float
    tier: ScorerTier
    detail: str = ""


class EvalManifestEntry(BaseModel):
    """One row of the checked-in eval plan (E09's exit artifact)."""

    component: str
    input_distribution: str
    expected_properties: list[str] = Field(default_factory=list)
    scorer: str
    regression_threshold: float
    level: PyramidLevel = "component"
    # The eval suite that measures this entry, when a suite does.
    suite: str | None = None


class EvalCase(BaseModel):
    """A single input/expectation pair executed by a pyramid runner."""

    case_id: str
    component: str
    input: Any
    expected: Any = None
    evidence: list[str] = Field(default_factory=list)


class EvalOutcome(BaseModel):
    """The result of running one case."""

    case_id: str
    component: str
    passed: bool
    score: float
    tier: ScorerTier
    detail: str = ""


class EvalReport(BaseModel):
    """Aggregate result of a pyramid level, including the NOLLM ratio."""

    level: PyramidLevel
    outcomes: list[EvalOutcome] = Field(default_factory=list)
    pass_rate: float = 0.0
    nollm_scorer_ratio: float = 0.0
    regression_threshold: float = 0.0

    @property
    def passed(self) -> bool:
        """Whether the level clears its configured regression threshold."""
        return self.pass_rate >= self.regression_threshold
