"""Hallucination-protection value types (LLD §10)."""

from typing import Literal

from pydantic import BaseModel, Field

# E26's differential diagnosis. The distinction is the whole point: a
# grounding failure is fixed by retrieval, a reasoning failure is not.
FailureKind = Literal["grounding", "reasoning", "none"]


class AtomicClaim(BaseModel):
    """One checkable assertion extracted from a generation.

    Scoring a paragraph as a unit hides the case that matters: three
    correct sentences and one fabricated number average out to "mostly
    right", and the fabricated number is what gets acted on.
    """

    text: str
    position: int = 0
    is_verifiable: bool = True


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
    escalated_to_human: bool = False


class GenerationAssessment(BaseModel):
    """Per-claim verdicts for a whole generation."""

    # Agreement across resamples, when that tier ran at all. `None`
    # means it did not: unmeasured is not the same as consistent.
    self_consistency: float | None = None
    verdicts: list[DetectionVerdict] = Field(default_factory=list)
    unsupported: list[str] = Field(default_factory=list)
    hallucination_rate: float = 0.0
    nollm_resolved_ratio: float = 1.0

    @property
    def clean(self) -> bool:
        """Whether every checkable claim was supported."""
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
