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
    EvidenceCheck,
    FailureKind,
    GenerationAssessment,
    TrustMeasurement,
    TrustReport,
)
from agentic_commons.llm.embeddings import tokenize
from agentic_commons.logging import get_logger, instrument, logged

# Sentence boundaries, and the connectives that join two claims into one
# sentence — both produce claims that must be checked separately.
#
# The semicolon belongs here for the same reason the full stop does. Left
# out, "the refund has not been initiated; the status shows none" stayed a
# single claim, and its two halves were scored against evidence that
# matched only one of them — so a sentence half of which was verbatim
# supported scored as entirely unsupported.
_SENTENCE = re.compile(r"(?<=[.!?;])\s+")
# Causal connectives split too. "Queued for review because the amount
# exceeds $50" is two claims from two tiers — the outcome from a tool
# result, the reason from the policy — and no single premise entails both.
# "Totals $117.23, so I cannot issue a $1 refund" likewise: joined, NLI
# read the pair as contradicting "a refund is always for the full total".
_CONJUNCTION = re.compile(r",?\s+\b(?:and|but|however|although|while|because|since|so)\b\s+")

# `Chunk.metadata` key for the order in which an observation was made this
# turn. Same-tier evidence that disagrees is settled by the later one.
OBSERVATION_SEQUENCE = "observation_sequence"

# A claim with no content words is not checkable ("Sure!", "Let me know.").
_MIN_CONTENT_TOKENS = 3

# The concrete details a reply can only get from the record: identifiers,
# amounts, dates, weekdays. They are what separates a claim the detector
# merely could not score from one that asserts something the evidence
# does not contain — "the refund status shows none" against a line that
# says exactly that, versus "your package will arrive on Tuesday" against
# a line that mentions no day at all. The first is an abstention worth
# tolerating; the second is a fabrication wearing the same score.
_SPECIFIC = re.compile(
    r"\d[\d,.:/-]*"
    r"|\b(?:mon|tues|wednes|thurs|fri|satur|sun)day\b"
    r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b",
    re.IGNORECASE,
)


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


def _novel_specifics(claim: str, retrieved: list[Chunk]) -> list[str]:
    """Concrete details the claim states and the evidence never mentions."""
    evidence = " ".join(chunk.content for chunk in retrieved).casefold()
    found = {value.casefold().rstrip(".,:") for value in _SPECIFIC.findall(claim)}
    return sorted(value for value in found if value and value not in evidence)


def _later_observation(
    first: tuple[EntailmentResult, Chunk],
    second: tuple[EntailmentResult, Chunk],
) -> tuple[EntailmentResult, Chunk] | None:
    """Whichever of two evidence items was observed later, when both say when."""
    a = first[1].metadata.get(OBSERVATION_SEQUENCE)
    b = second[1].metadata.get(OBSERVATION_SEQUENCE)
    if not isinstance(a, int) or not isinstance(b, int) or a == b:
        return None
    return first if a > b else second


class ClaimDecomposer:
    """Breaks a generation into atomic, separately checkable claims."""

    def decompose(self, generation: str) -> list[AtomicClaim]:
        """Split on sentence boundaries, then on joining conjunctions."""
        claims: list[AtomicClaim] = []
        for sentence in _SENTENCE.split(generation.strip()):
            for part in _CONJUNCTION.split(sentence):
                text = part.strip().rstrip(".;")
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
        self._floor = settings.evidence_min_similarity
        self._authority = settings.evidence_authority
        self._support_only = frozenset(settings.support_only_tiers)
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

        relevant = [chunk for chunk in retrieved if chunk.score >= self._floor]
        results = list(
            await asyncio.gather(
                *(self._entailment.entails(chunk.content, claim) for chunk in relevant)
            )
        )
        verdict = (
            self._combine(claim, relevant, results)
            if relevant
            else self._nothing_relevant(claim, retrieved)
        )
        return verdict.model_copy(
            update={
                "evidence_checks": self._evidence_checks(retrieved, dict(zip(map(id, relevant), results))),
                "threshold": self._threshold if any(r.calibrated for r in results) else None,
            }
        )

    def _nothing_relevant(self, claim: str, retrieved: list[Chunk]) -> DetectionVerdict:
        """No evidence is close enough to judge the claim by.

        That is an abstention, not a refutation — unless the claim brings
        a specific (an amount, an id) that no evidence contains, which is
        unsourced whatever NLI would have said.
        """
        novel = _novel_specifics(claim, retrieved)
        return DetectionVerdict(
            claim=claim,
            supported=False,
            undetermined=not novel,
            confidence=0.0,
            tier="nli_entailment",
            rationale=(
                f"{', '.join(novel)} appear(s) in the claim but in no evidence"
                if novel
                else f"no evidence is related to this claim (all below similarity {self._floor})"
            ),
            detector_model=self._detector_model,
            retrieval_top_k=self._top_k,
            retrieval_method=self._retrieval_method,
            retrieved_evidence=[chunk.content for chunk in retrieved],
        )

    def _evidence_checks(
        self, retrieved: list[Chunk], judged: dict[int, EntailmentResult]
    ) -> list[EvidenceCheck]:
        """Every retrieved item in similarity order, with the NLI result it got (if any)."""
        checks = []
        for rank, chunk in enumerate(retrieved, start=1):
            result = judged.get(id(chunk))
            checks.append(
                EvidenceCheck(
                    rank=rank,
                    similarity=round(chunk.score, 3),
                    tier=chunk.kind,
                    excerpt=chunk.content,
                    nli_label=result.label if result else f"not checked (below {self._floor})",
                    nli_score=round(result.score, 3) if result else 0.0,
                    decisive=result is not None
                    and result.label != "unknown"
                    and self._is_decisive(result),
                )
            )
        return checks

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
            if result.label == "contradiction" and chunk.kind not in self._support_only
        ]

        best_entailment = max(entailed, key=lambda item: item[0].score, default=None)
        best_contradiction = max(contradicted, key=lambda item: item[0].score, default=None)

        # Decisive evidence on both sides: the more authoritative tier wins
        # (a tool result on what happened beats a policy paragraph); only a
        # disagreement between equals is unsafe to collapse into PASS.
        if (
            best_entailment
            and best_contradiction
            and self._is_decisive(best_entailment[0])
            and self._is_decisive(best_contradiction[0])
        ):
            return self._conflict(claim, retrieved, best_entailment, best_contradiction)

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
                undetermined=True,
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
        # Nothing here refuted the claim. A neutral label, or an
        # entailment too weak to count, means the detector abstained —
        # and an abstention from a token-overlap heuristic is not
        # evidence that a tool-grounded sentence is a hallucination. It
        # was, though: "unknown" reached the guardrail as `supported=False`
        # and silenced a correct reply behind a generic apology.
        #
        # An abstention is only safe where the claim adds no detail of its
        # own. One that introduces an amount, a date or an identifier the
        # evidence never mentions is not unscored, it is unsourced — the
        # detector's silence says nothing, and the specific says plenty.
        novel = _novel_specifics(claim, retrieved)
        if novel:
            return self._verdict(
                claim,
                retrieved,
                result,
                supported=False,
                rationale=(
                    f"{', '.join(novel)} appear(s) in the claim but in no retrieved evidence"
                ),
            )
        return self._verdict(claim, retrieved, result, supported=False, undetermined=True)

    def _conflict(
        self,
        claim: str,
        retrieved: list[Chunk],
        entails: tuple[EntailmentResult, Chunk],
        contradicts: tuple[EntailmentResult, Chunk],
    ) -> DetectionVerdict:
        """Resolve entailment vs contradiction by the authority of their evidence."""
        for_rank = self._authority.get(entails[1].kind, 0)
        against_rank = self._authority.get(contradicts[1].kind, 0)
        later = _later_observation(entails, contradicts)
        if for_rank == against_rank and later is not None:
            # Two observations of the same tier disagreeing are usually one
            # state read before an action and one after it. The world is
            # what the last observation saw.
            return self._verdict(
                claim,
                retrieved,
                later[0],
                supported=later is entails,
                rationale=f"the later {later[1].kind} observation ({later[0].label}) supersedes the earlier one",
            )
        if for_rank != against_rank:
            winner, loser = (entails, contradicts) if for_rank > against_rank else (contradicts, entails)
            return self._verdict(
                claim,
                retrieved,
                winner[0],
                supported=winner is entails,
                rationale=(
                    f"{winner[1].kind} evidence {winner[0].label} outranks "
                    f"{loser[1].kind} evidence {loser[0].label}"
                ),
            )
        return self._verdict(
            claim,
            retrieved,
            entails[0],
            supported=False,
            rationale=(
                f"conflicting {entails[1].kind} evidence: one item entails the claim "
                "while another of equal authority contradicts it"
            ),
        )

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
        undetermined: bool = False,
    ) -> DetectionVerdict:
        """Build an auditable verdict with the exact evidence sent to NLI."""
        detail = rationale or result.rationale
        if result.label == "unknown" and not rationale:
            detail = f"unknown: {result.rationale}"
        return DetectionVerdict(
            claim=claim,
            supported=supported,
            undetermined=undetermined,
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
