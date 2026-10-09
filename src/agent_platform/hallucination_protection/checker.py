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

from collections.abc import Awaitable, Callable
from typing import Protocol, runtime_checkable

from agent_platform.config import get_settings
from agent_platform.context.retrieval import RelevanceRanker
from agent_platform.context.types import Chunk
from agent_platform.evals.entailment import EntailmentModel, entailment_backend
from agent_platform.evals.types import Claim, Verdict
from agent_platform.hallucination_protection.claims import ClaimDecomposer
from agent_platform.hallucination_protection.resolver import GroundingResolver
from agent_platform.hallucination_protection.types import (
    DetectionVerdict,
    FailureKind,
    GenerationAssessment,
    TrustMeasurement,
    TrustReport,
)
from agent_platform.llm.embeddings import tokenize
from agent_platform.logging import get_logger, logged

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


class DetectionLadder:
    """Compatibility facade over the retrieval → NLI grounding verifier."""

    def __init__(
        self,
        cascade: RelevanceRanker,
        decomposer: ClaimDecomposer | None = None,
        entailment: EntailmentModel | None = None,
    ) -> None:
        self._resolver = GroundingResolver(cascade, entailment)
        self._decomposer = decomposer or ClaimDecomposer()
        self._logger = get_logger("hallucination_protection")

    @property
    def identity(self) -> tuple[str, str]:
        """(backend, model) of the entailment model that judges this ladder's claims."""
        model = self._resolver.entailment
        return entailment_backend(model), str(getattr(model, "model_name", ""))

    @logged("hallucination_protection")
    async def score(self, claim: str, evidence: list[Chunk]) -> DetectionVerdict:
        """Class the claim, then ask the tiers that may ground it."""
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
        verdicts = [
            (await self._resolver.resolve(claim.text, evidence, claim.claim_class)).model_copy(
                update={"sentence": claim.sentence}
            )
            for claim in claims
        ]
        agreement = await self._agreement(verdicts, generation, resample)
        unsupported = [
            verdict.claim
            for verdict in verdicts
            if not verdict.supported and not verdict.undetermined
        ]
        undetermined = [verdict.claim for verdict in verdicts if verdict.undetermined]
        return GenerationAssessment(
            self_consistency=agreement,
            verdicts=verdicts,
            unsupported=unsupported,
            undetermined=undetermined,
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
        if not any(not verdict.supported and not verdict.reason_code for verdict in verdicts):
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
