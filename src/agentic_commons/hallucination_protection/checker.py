"""Hallucination detection, diagnosis and trust calibration (E25-E27, LLD §10).

The detection ladder is not a new cascade — it *is* `evals`'s NOLLM
cascade, injected. Confidence, then entropy, then entailment, then a
judge, then a human. Building a second ladder here would mean two places
to keep calibrated and two values of `nollm_scorer_ratio`.

The part teams skip is diagnosis. "The model hallucinated" is not a
finding; it names a symptom with two unrelated causes. If the evidence
was never retrieved, no amount of prompt work fixes it. If the evidence
was right there and the model concluded otherwise, more retrieval makes
it worse.
"""

import re
from typing import Protocol, runtime_checkable

from agentic_commons.config import get_settings
from agentic_commons.context.types import Chunk
from agentic_commons.evals.types import Claim, Verdict
from agentic_commons.hallucination_protection.types import (
    AtomicClaim,
    DetectionVerdict,
    FailureKind,
    GenerationAssessment,
    TrustMeasurement,
    TrustReport,
)
from agentic_commons.llm.embeddings import tokenize
from agentic_commons.logging import get_logger, logged

# Sentence boundaries, and the connectives that join two claims into one
# sentence — both produce claims that must be checked separately.
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_CONJUNCTION = re.compile(r",?\s+\b(?:and|but|however|although|while)\b\s+")

# A claim with no content words is not checkable ("Sure!", "Let me know.").
_MIN_CONTENT_TOKENS = 3


@runtime_checkable
class ClaimResolver(Protocol):
    """Resolves a claim against evidence.

    Structurally satisfied by `evals.NollmScoringCascadeImpl`. Injected
    rather than constructed so the detection ladder shares the *same*
    cascade instance as the rest of the system, and therefore the same
    calibration and the same `nollm_scorer_ratio`.
    """

    async def resolve(self, claim: Claim) -> Verdict: ...

    @property
    def nollm_scorer_ratio(self) -> float: ...


@runtime_checkable
class HallucinationDetector(Protocol):
    """Scores whether a claim is supported by its evidence."""

    async def score(self, claim: str, evidence: list[Chunk]) -> DetectionVerdict: ...


class ClaimDecomposer:
    """Breaks a generation into atomic, separately checkable claims."""

    def decompose(self, generation: str) -> list[AtomicClaim]:
        """Split on sentence boundaries, then on joining conjunctions."""
        claims: list[AtomicClaim] = []
        for sentence in _SENTENCE.split(generation.strip()):
            for part in _CONJUNCTION.split(sentence):
                text = part.strip().rstrip(".")
                if not text:
                    continue
                claims.append(
                    AtomicClaim(
                        text=text,
                        position=len(claims),
                        is_verifiable=len(tokenize(text)) >= _MIN_CONTENT_TOKENS,
                    )
                )
        return claims


class DetectionLadder:
    """The detection ladder, delegating every tier to the shared cascade."""

    def __init__(self, cascade: ClaimResolver, decomposer: ClaimDecomposer | None = None) -> None:
        settings = get_settings().hallucination_protection
        self._cascade = cascade
        self._decomposer = decomposer or ClaimDecomposer()
        self._confidence_threshold = settings.confidence_threshold
        self._enabled = settings.enabled
        self._logger = get_logger("hallucination_protection")

    @logged("hallucination_protection")
    async def score(self, claim: str, evidence: list[Chunk]) -> DetectionVerdict:
        """Resolve one claim against its evidence through the cascade."""
        verdict = await self._cascade.resolve(
            Claim(text=claim, evidence=[chunk.content for chunk in evidence])
        )
        return DetectionVerdict(
            claim=claim,
            supported=verdict.label == "supported",
            confidence=verdict.score,
            tier=verdict.tier,
            rationale=verdict.rationale,
            escalated_to_human=verdict.label == "deferred",
        )

    async def assess(self, generation: str, evidence: list[Chunk]) -> GenerationAssessment:
        """Check every verifiable claim in a generation."""
        if not self._enabled:
            return GenerationAssessment()
        claims = [claim for claim in self._decomposer.decompose(generation) if claim.is_verifiable]
        verdicts = [await self.score(claim.text, evidence) for claim in claims]
        unsupported = [verdict.claim for verdict in verdicts if not verdict.supported]
        return GenerationAssessment(
            verdicts=verdicts,
            unsupported=unsupported,
            hallucination_rate=len(unsupported) / len(verdicts) if verdicts else 0.0,
            nollm_resolved_ratio=self._cascade.nollm_scorer_ratio,
        )

    def consistency(self, samples: list[str]) -> float:
        """Agreement across repeated samples of the same question.

        Self-consistency is a cheap, model-only signal: a fabricated
        detail varies between samples, while a grounded one does not. It
        needs no evidence, which is why it sits high in the ladder.
        """
        if len(samples) < 2:
            return 1.0
        token_sets = [set(tokenize(sample)) for sample in samples]
        pairs = [
            len(left & right) / len(left | right)
            for index, left in enumerate(token_sets)
            for right in token_sets[index + 1 :]
            if left | right
        ]
        return sum(pairs) / len(pairs) if pairs else 1.0


class GroundingReasoningDiagnoser:
    """Was the evidence missing, or present and ignored? (E26).

    One symptom, two causes, opposite fixes. Routing every hallucination
    to "improve retrieval" is why some of them never get fixed.
    """

    def __init__(self) -> None:
        self._overlap_threshold = (
            get_settings().hallucination_protection.grounding_overlap_threshold
        )
        self._logger = get_logger("hallucination_protection")

    def diagnose(self, claim: str, retrieved: list[Chunk]) -> FailureKind:
        """Classify an unsupported claim by whether its evidence was present."""
        if not retrieved:
            return "grounding"
        best = max((self.overlap(claim, chunk.content) for chunk in retrieved), default=0.0)
        kind: FailureKind = "reasoning" if best >= self._overlap_threshold else "grounding"
        self._logger.debug("Diagnosed failure", kind=kind, best_overlap=round(best, 3))
        return kind

    @staticmethod
    def overlap(claim: str, evidence: str) -> float:
        """Fraction of the claim's content tokens present in the evidence."""
        claim_tokens = set(tokenize(claim))
        if not claim_tokens:
            return 0.0
        return len(claim_tokens & set(tokenize(evidence))) / len(claim_tokens)


class TrustCalibrationTracker:
    """Measures whether humans trust the agent about as much as they should.

    Two opposite failures, both expensive. Automation bias is accepting a
    wrong answer; algorithm aversion is rejecting a right one. A review
    step that does either consistently is not a control, and only
    measurement distinguishes them.
    """

    def __init__(self) -> None:
        self._measurements: list[TrustMeasurement] = []

    def record(self, agent_correct: bool, human_accepted: bool) -> None:
        """Record one human decision about one agent output."""
        self._measurements.append(
            TrustMeasurement(agent_correct=agent_correct, human_accepted=human_accepted)
        )

    def report(self) -> TrustReport:
        """Bias, aversion and appropriate-reliance rates over all samples."""
        total = len(self._measurements)
        if not total:
            return TrustReport(
                samples=0,
                automation_bias_rate=0.0,
                algorithm_aversion_rate=0.0,
                appropriate_reliance_rate=0.0,
            )
        bias = sum(not m.agent_correct and m.human_accepted for m in self._measurements)
        aversion = sum(m.agent_correct and not m.human_accepted for m in self._measurements)
        return TrustReport(
            samples=total,
            automation_bias_rate=bias / total,
            algorithm_aversion_rate=aversion / total,
            appropriate_reliance_rate=(total - bias - aversion) / total,
        )
