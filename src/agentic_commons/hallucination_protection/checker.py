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

import asyncio
import re
from collections.abc import Awaitable, Callable
from typing import Protocol, runtime_checkable

from agentic_commons.config import get_settings
from agentic_commons.context.retrieval import DenseRetriever, RelevanceRanker
from agentic_commons.context.types import Chunk
from agentic_commons.evals.entailment import EntailmentResult, create_entailment_model
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
from agentic_commons.logging import get_logger, instrument, logged

# Sentence boundaries, and the connectives that join two claims into one
# sentence — both produce claims that must be checked separately.
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_CONJUNCTION = re.compile(r",?\s+\b(?:and|but|however|although|while)\b\s+")

# A claim with no content words is not checkable ("Sure!", "Let me know.").
_MIN_CONTENT_TOKENS = 3


# Asks the same question again, n more times. Injected rather than held,
# because the detector must not own an LLM client: a checker that can
# call a model on its own is a checker with its own opinions.
Resampler = Callable[[int], Awaitable[list[str]]]


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


@instrument("hallucination_protection")
class GroundingResolver:
    """Retrieve relevant evidence, then verify it with NLI.

    Embeddings have one job here: retrieve the best evidence for a claim.
    They never decide whether the claim is true. NLI makes that decision.
    This separation lets the same DenseRetriever be reused by future RAG
    flows without turning semantic similarity into a truth signal.
    """

    def __init__(
        self,
        ranker: RelevanceRanker,
        entailment=None,
    ) -> None:
        settings = get_settings().hallucination_protection
        self._retriever = DenseRetriever(ranker)
        self._top_k = settings.embedding_top_k
        self._threshold = settings.nli_entailment_threshold
        self._entailment = entailment or create_entailment_model()
        self._logger = get_logger("hallucination_protection")

    async def resolve(self, claim: str, evidence: list[Chunk]) -> DetectionVerdict:
        """Resolve one claim using top-k retrieval followed by NLI."""
        self._retriever.index(evidence)
        retrieved = await self._retriever.retrieve(claim, self._top_k)
        if not retrieved:
            return DetectionVerdict(
                claim=claim,
                supported=False,
                confidence=0.0,
                tier="nli_entailment",
                rationale="no evidence was retrieved for the claim",
                detector_model=self._detector_model,
                retrieval_top_k=self._top_k,
                retrieval_method=self._retrieval_method,
            )

        results = await asyncio.gather(
            *(self._entailment.entails(chunk.content, claim) for chunk in retrieved)
        )
        return self._combine(claim, retrieved, results)

    @property
    def _retrieval_method(self) -> str:
        """Report the actual retrieval path used by the injected ranker."""
        cascade = getattr(self._retriever, "_ranker", None)
        enabled = getattr(get_settings().evals.cascade, "embedding_similarity", False)
        return "embedding" if enabled and cascade is not None else "lexical_fallback"

    @property
    def _detector_model(self) -> str:
        """Expose the concrete NLI model, including its deterministic fallback."""
        return str(getattr(self._entailment, "model_name", "")) or (
            "lexical-entailment-fallback"
        )

    def _combine(
        self,
        claim: str,
        retrieved: list[Chunk],
        results: list[EntailmentResult],
    ) -> DetectionVerdict:
        """Collapse per-chunk NLI decisions without treating similarity as truth."""
        entailed = [
            (result, chunk)
            for result, chunk in zip(results, retrieved)
            if result.label == "entailment"
        ]
        contradicted = [
            (result, chunk)
            for result, chunk in zip(results, retrieved)
            if result.label == "contradiction"
        ]

        best_entailment = max(entailed, key=lambda item: item[0].score, default=None)
        best_contradiction = max(contradicted, key=lambda item: item[0].score, default=None)

        # Conflicting high-confidence evidence is unsafe to collapse into PASS.
        if (
            best_entailment
            and best_contradiction
            and self._is_decisive(best_entailment[0])
            and self._is_decisive(best_contradiction[0])
        ):
            result, _ = best_entailment
            return self._verdict(
                claim,
                retrieved,
                result,
                supported=False,
                rationale=(
                    "conflicting evidence: one retrieved chunk entails the claim "
                    "while another contradicts it"
                ),
            )

        if best_entailment and self._is_decisive(best_entailment[0]):
            result, _ = best_entailment
            return self._verdict(claim, retrieved, result, supported=True)

        if best_contradiction and self._is_decisive(best_contradiction[0]):
            result, _ = best_contradiction
            return self._verdict(claim, retrieved, result, supported=False)

        # Neutral/low-confidence NLI is UNKNOWN, not contradiction. If the
        # model selected entailment but its probability is below the grounding
        # threshold, retain that fact in the audit trail so the TUI explains
        # why an apparently matching claim was rejected.
        result = max(results, key=lambda item: item.score, default=None)
        if result is None:
            return self._verdict(
                claim,
                retrieved,
                EntailmentResult(
                    label="unknown",
                    entailed=False,
                    score=0.0,
                    rationale="NLI returned no result",
                    model=self._detector_model,
                ),
                supported=False,
            )
        if result.label == "entailment":
            # Name the detector that actually decided. Saying "NLI" when the
            # lexical fallback ran sent a reader looking for a model problem
            # that was not there.
            result = result.model_copy(
                update={
                    "rationale": (
                        f"{result.model or 'the detector'} predicted entailment "
                        f"({result.score:.3f}) but confidence is below the configured "
                        f"threshold ({self._threshold:.3f})"
                    )
                }
            )
        return self._verdict(claim, retrieved, result, supported=False)

    def _is_decisive(self, result: EntailmentResult) -> bool:
        """Whether a result settles the claim, judged on its own scale.

        `nli_entailment_threshold` is a probability, and only a calibrated
        classifier produces one. The lexical fallback scores token
        coverage instead, and a claim that restates its evidence exactly
        still covers only the words the evidence happens to contain —
        "A refund has already been approved for order 10864" covers four
        of nine against "Order 10864 refund status: approved". Comparing
        that ratio to 0.60 rejected true claims and escalated correct
        answers to a human.

        So the threshold applies only where it means something. An
        uncalibrated model's label is its decision: it has already gated
        on `evals.nli.min_overlap` and checked negation polarity, and
        there is no second, comparable number to test.
        """
        return result.score >= self._threshold if result.calibrated else True

    def _verdict(
        self,
        claim: str,
        retrieved: list[Chunk],
        result: EntailmentResult,
        *,
        supported: bool,
        rationale: str | None = None,
    ) -> DetectionVerdict:
        """Build an auditable verdict with the exact evidence sent to NLI."""
        detail = rationale or result.rationale
        if result.label == "unknown" and not rationale:
            detail = f"unknown: {result.rationale}"
        return DetectionVerdict(
            claim=claim,
            supported=supported,
            confidence=result.score,
            tier="nli_entailment",
            rationale=detail,
            detector_model=result.model or self._detector_model,
            retrieval_top_k=self._top_k,
            retrieval_method=self._retrieval_method,
            retrieved_evidence=[chunk.content for chunk in retrieved],
        )


class DetectionLadder:
    """Compatibility facade over the retrieval → NLI grounding verifier."""

    def __init__(self, cascade: ClaimResolver, decomposer: ClaimDecomposer | None = None) -> None:
        self._resolver = GroundingResolver(cascade)
        self._decomposer = decomposer or ClaimDecomposer()
        self._logger = get_logger("hallucination_protection")

    @logged("hallucination_protection")
    async def score(self, claim: str, evidence: list[Chunk]) -> DetectionVerdict:
        """Retrieve relevant evidence and ask NLI whether it entails the claim."""
        return await self._resolver.resolve(claim, evidence)

    async def assess(
        self,
        generation: str,
        evidence: list[Chunk],
        resample: Resampler | None = None,
    ) -> GenerationAssessment:
        """Check every verifiable claim independently.

        `resample` is the next rung of the ladder, and is used only when
        the entailment tier could not decide — which in practice means the
        hosted NLI was unreachable and the lexical fallback answered on a
        scale that is not a probability. Comparing that score against a
        probability threshold is what previously blocked true claims as
        hallucinations.

        Asking the model the same question again costs completions, so it
        happens on the rare inconclusive path rather than every turn. A
        detector that tripled the cost of an ordinary reply is a detector
        that gets switched off.
        """
        claims = [claim for claim in self._decomposer.decompose(generation) if claim.is_verifiable]
        verdicts = [await self.score(claim.text, evidence) for claim in claims]
        agreement = await self._agreement(verdicts, generation, resample)
        unsupported = [verdict.claim for verdict in verdicts if not verdict.supported]
        return GenerationAssessment(
            self_consistency=agreement,
            verdicts=verdicts,
            unsupported=unsupported,
            hallucination_rate=len(unsupported) / len(verdicts) if verdicts else 0.0,
            # This verifier is deliberately model-based. The embedding
            # retriever is retrieval infrastructure, not a NOLLM truth tier.
            nollm_resolved_ratio=0.0,
        )

    async def _agreement(
        self,
        verdicts: list[DetectionVerdict],
        generation: str,
        resample: Resampler | None,
    ) -> float | None:
        """Self-consistency, measured only where it can change the answer.

        `None` means the tier did not run: either nothing was
        inconclusive, or no resampler was wired. That is deliberately not
        1.0 — "unmeasured" and "perfectly consistent" are different
        claims, and reporting the second for the first is how a detector
        comes to look like it is working.
        """
        settings = get_settings().hallucination_protection
        if resample is None or not settings.consistency_enabled:
            return None
        if not any(verdict.tier == "nli_entailment" and not verdict.supported for verdict in verdicts):
            return None
        samples = [generation, *await resample(settings.consistency_samples - 1)]
        agreement = self.consistency(samples)
        self._logger.info(
            "Self-consistency checked after an inconclusive entailment verdict",
            samples=len(samples),
            agreement=round(agreement, 3),
            threshold=settings.consistency_agreement_threshold,
            stable=agreement >= settings.consistency_agreement_threshold,
        )
        return agreement

    def consistency(self, samples: list[str]) -> float:
        """Agreement across repeated samples of the same question."""
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
