"""Hallucination-protection value types (LLD §10)."""

from typing import Literal

from pydantic import BaseModel, Field

# E26's differential diagnosis. The distinction is the whole point: a
# grounding failure is fixed by retrieval, a reasoning failure is not.
FailureKind = Literal["grounding", "reasoning", "none"]

# What a claim is, which decides what may ground it: a fact about this case
# (only a record can), a rule or a promise (the business's own statements),
# or manner of speaking ("Thanks for reaching out"), which is not checked.
ClaimClass = Literal["record", "rule", "conversational"]


class AtomicClaim(BaseModel):
    """One checkable assertion extracted from a generation.

    Scoring a paragraph as a unit hides the case that matters: three
    correct sentences and one fabricated number average out to "mostly
    right", and the fabricated number is what gets acted on.
    """

    text: str
    position: int = 0
    # Which sentence of the reply this claim came from. A claim is finer
    # than a sentence and may be restated (a carved rule), so what a claim
    # removes is found by this index, never by matching its text.
    sentence: int = 0
    is_verifiable: bool = True
    claim_class: ClaimClass = "record"


class EvidenceCheck(BaseModel):
    """One retrieved evidence item for a claim: how it was found, and what NLI said.

    Kept for every claim, supported or not, so a passing verdict can be
    checked as closely as a failing one: which evidence retrieval surfaced
    (by cosine similarity), and which of it the entailment model agreed with.
    """

    rank: int
    similarity: float
    tier: str
    excerpt: str
    nli_label: str
    nli_score: float
    # Whether this result was strong enough to settle the claim on its own.
    decisive: bool
    # Where the line came from (`Chunk.source`): a page's URL, a tool's name, an article.
    source: str = ""
    # The model that read this line (the NLI model, or the second opinion); "" for a line of
    # the record read directly.
    model: str = ""


class DetectionVerdict(BaseModel):
    """Whether a claim is supported, with the evidence and detector behind it."""

    claim: str
    supported: bool
    confidence: float
    tier: str = ""
    rationale: str = ""
    detector_model: str = ""
    retrieval_top_k: int = 0
    retrieval_method: str = ""
    retrieved_evidence: list[str] = Field(default_factory=list)
    # The full trail: every retrieved item with its similarity and NLI result.
    evidence_checks: list[EvidenceCheck] = Field(default_factory=list)
    # The bar a calibrated NLI score had to clear; None for an uncalibrated
    # (lexical) detector, whose label is its decision.
    threshold: float | None = None
    escalated_to_human: bool = False
    claim_class: str = ""
    sentence: int = -1
    # Set when a deterministic check refuted the claim: what it found, in
    # the code the card and the audit trail carry.
    reason_code: str = ""
    # The detector abstained rather than refuting. `supported=False` alone
    # conflates "the evidence says otherwise" with "I could not tell", and
    # only the first is a hallucination.
    undetermined: bool = False


class GenerationAssessment(BaseModel):
    """Per-claim verdicts for a whole generation."""

    # Agreement across resamples, when that tier ran at all. `None`
    # means it did not: unmeasured is not the same as consistent.
    self_consistency: float | None = None
    verdicts: list[DetectionVerdict] = Field(default_factory=list)
    unsupported: list[str] = Field(default_factory=list)
    # Claims the detector could not settle either way. Reported so the
    # trace shows what went unchecked, but never grounds for blocking a
    # reply: an abstention is not a finding.
    undetermined: list[str] = Field(default_factory=list)
    hallucination_rate: float = 0.0
    nollm_resolved_ratio: float = 1.0

    @property
    def clean(self) -> bool:
        """Whether any checkable claim was actually refuted."""
        return not self.unsupported


class TrustMeasurement(BaseModel):
    """One observation of how a human treated an agent's output (E27)."""

    agent_correct: bool
    human_accepted: bool


class TrustReport(BaseModel):
    """Automation bias and algorithm aversion, measured rather than assumed."""

    samples: int
    automation_bias_rate: float
    algorithm_aversion_rate: float
    appropriate_reliance_rate: float
