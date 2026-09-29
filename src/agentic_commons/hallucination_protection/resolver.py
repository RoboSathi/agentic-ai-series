"""Grounding: which tiers may ground a claim, and what each says.

The evidence is not one pool. A claim is classed (record, rule or
conversation) and is then read against only the tiers that may ground its
class, each with its own top-k, so a policy section can no longer crowd a
tool fact out of the ranking. That is what removed the reconciliation
machinery this module used to carry — authority ranks, support-only
tiers, supersession, one voice per observation, a joined premise —
all of it existed to undo the pooling.

Embeddings have one job here: retrieve the best evidence for a claim.
They never decide whether the claim is true. NLI makes that decision.
"""

import asyncio
import re

from agentic_commons.config import get_settings
from agentic_commons.config.settings import GroundingTierSettings
from agentic_commons.context.retrieval import DenseRetriever, RelevanceRanker
from agentic_commons.context.types import Chunk
from agentic_commons.evals.entailment import (
    EntailmentModel,
    EntailmentResult,
    create_entailment_model,
)
from agentic_commons.hallucination_protection.claims import ClaimClassifier
from agentic_commons.hallucination_protection.direct import DirectChecker, DirectFinding, numbers
from agentic_commons.hallucination_protection.types import (
    ClaimClass,
    DetectionVerdict,
    EvidenceCheck,
)
from agentic_commons.llm.embeddings import tokenize
from agentic_commons.logging import instrument

# `Chunk.metadata` key for the order in which an observation was made this
# turn. Same-tier evidence that disagrees is settled by the later one.
OBSERVATION_SEQUENCE = "observation_sequence"
# `Chunk.metadata` key marking an observation that changed something (a
# write). A read taken before a write of the same thing describes a world
# that no longer exists.
STATE_CHANGING = "state_changing"
# The detector name of a rule that restates the claim in nearly its own words.
RESTATEMENT_MODEL = "rule-restatement"
# The label a line of the record carries when it was read directly and not
# scored by a model.
RECORD_LABEL = "record"
# What decided a claim with no evidence tier: the direct check, or nothing.
DIRECT_TIER = "deterministic"
NO_TIER = "none"

_DECIDING_LABELS = frozenset({"entailment", "contradiction"})
# The NLI model's third label, as a reader should meet it: not a failure.
_LABEL_NAMES = {"unknown": "neutral"}


def _outvotes(item: tuple[EntailmentResult, Chunk], held: tuple[EntailmentResult, Chunk]) -> bool:
    """Whether a decisive result should speak for its observation instead of the one held."""
    if item[0].label != held[0].label:
        return item[0].label == "entailment"
    return item[0].score > held[0].score


class _TierOutcome:
    """What one tier concluded about one claim."""

    def __init__(
        self,
        tier: GroundingTierSettings,
        label: str,
        result: EntailmentResult | None,
        checks: list[EvidenceCheck],
        retrieved: list[Chunk],
        note: str = "",
    ) -> None:
        self.note = note
        self.tier = tier
        self.label = label
        self.result = result
        self.checks = checks
        self.retrieved = retrieved


@instrument("hallucination_protection")
class GroundingResolver:
    """Resolve one claim against the tiers that may ground it."""

    def __init__(
        self,
        ranker: RelevanceRanker,
        entailment: EntailmentModel | None = None,
        classifier: ClaimClassifier | None = None,
    ) -> None:
        settings = get_settings().hallucination_protection
        self._retriever = DenseRetriever(ranker)
        self._top_k = settings.embedding_top_k
        self._threshold = settings.nli_entailment_threshold
        self._floor = settings.evidence_min_similarity
        self._tiers = settings.tiers
        self._restatement = settings.rule_restatement_coverage
        self._identifier = (
            re.compile(settings.refute_identifier_pattern)
            if settings.refute_identifier_pattern
            else None
        )
        self._negations = tuple(m.casefold() for m in get_settings().evals.nli.negation_markers)
        self._entailment = entailment or create_entailment_model()
        self._classifier = classifier or ClaimClassifier()
        self._direct = DirectChecker()
        # A state record ("Order N refund status: X") is read by the state
        # model. NLI cannot: it took `pending_review` to contradict "submitted".
        self._state_line = re.compile(settings.refund_state_pattern, re.IGNORECASE)
        self._graded_kinds = frozenset(kind for tier in self._tiers for kind in tier.kinds)
        self._record_kinds = frozenset(
            kind for tier in self._tiers if "record" in tier.claims for kind in tier.kinds
        )

    async def resolve(
        self, claim: str, evidence: list[Chunk], claim_class: ClaimClass | None = None
    ) -> DetectionVerdict:
        """The verdict on one claim: refuted, supported, or undetermined."""
        kind = claim_class or self._classifier.classify(claim)
        if kind == "conversational":
            return self._verdict(
                claim,
                kind,
                NO_TIER,
                None,
                [],
                undetermined=True,
                rationale="conversation, not a claim",
            )
        evidence = self._current(evidence)
        finding = self._direct_finding(claim, kind, evidence)
        if finding is not None:
            return self._read(
                claim,
                kind,
                finding.chunk,
                supported=False,
                rationale=finding.rationale,
                reason_code=finding.reason_code,
            )
        record = [chunk for chunk in evidence if chunk.kind in self._record_kinds]
        settled = self._settled_by_record(claim, kind, record)
        if settled is not None:
            return self._read(claim, kind, settled[1], supported=True, rationale=settled[0])
        outcomes = [
            await self._judge(claim, tier, [c for c in evidence if c.kind in tier.kinds])
            for tier in self._tiers
            if kind in tier.claims
        ]
        return self._settle(claim, kind, outcomes)

    def _settled_by_record(
        self, claim: str, kind: ClaimClass, record: list[Chunk]
    ) -> tuple[str, Chunk] | None:
        """A record claim the record decides by itself: a refusal it shows, or a status it holds."""
        if kind != "record":
            return None
        return self._direct.refusal_support(claim, record) or self._direct.state_support(
            claim, record
        )

    def _read(
        self,
        claim: str,
        kind: ClaimClass,
        chunk: Chunk | None,
        *,
        supported: bool,
        rationale: str,
        reason_code: str = "",
    ) -> DetectionVerdict:
        """A verdict settled by reading the record, quoting the line it read."""
        checks = [] if chunk is None else [self._record_check(chunk)]
        return self._verdict(
            claim,
            kind,
            DIRECT_TIER,
            None,
            [] if chunk is None else [chunk],
            supported=supported,
            rationale=rationale,
            reason_code=reason_code,
            checks=checks,
        )

    @staticmethod
    def _record_check(chunk: Chunk) -> EvidenceCheck:
        return EvidenceCheck(
            rank=1,
            similarity=1.0,
            tier=chunk.kind,
            excerpt=chunk.content,
            nli_label=RECORD_LABEL,
            nli_score=1.0,
            decisive=True,
        )

    def _direct_finding(
        self, claim: str, kind: ClaimClass, evidence: list[Chunk]
    ) -> DirectFinding | None:
        """What the record, or the evidence as a whole, rules out without asking a model."""
        if kind == "record":
            return self._direct.refutation(
                claim, [chunk for chunk in evidence if chunk.kind in self._record_kinds]
            )
        return self._direct.unsourced_detail(
            claim, [chunk for chunk in evidence if chunk.kind in self._graded_kinds]
        )

    def _settle(
        self, claim: str, kind: ClaimClass, outcomes: list[_TierOutcome]
    ) -> DetectionVerdict:
        """A refutation that may stand wins; else the first support; else no verdict."""
        checks = self._ranked([check for outcome in outcomes for check in outcome.checks])
        retrieved = [chunk for outcome in outcomes for chunk in outcome.retrieved]
        for label, supported in (("contradiction", False), ("entailment", True)):
            for outcome in outcomes:
                if outcome.label == label and outcome.result is not None:
                    return self._verdict(
                        claim,
                        kind,
                        outcome.tier.name,
                        outcome.result,
                        retrieved,
                        supported=supported,
                        checks=checks,
                    )
        return self._verdict(
            claim,
            kind,
            NO_TIER,
            None,
            retrieved,
            undetermined=True,
            checks=checks,
            rationale="no tier decided: "
            + " · ".join(f"{outcome.tier.name}: {outcome.note}" for outcome in outcomes),
        )

    def _current(self, evidence: list[Chunk]) -> list[Chunk]:
        """The evidence with every read the turn has since overtaken taken out.

        "No refund has been requested for order 10169" was true when it was
        read and false once the refund was referred, and nothing the later
        observation says needs to contradict it for that to be so. A read
        is overtaken when a *later* write names the same order; a read
        about another order, or in a turn that wrote nothing, stands.
        """
        writes = [
            (sequence, numbers(chunk.content))
            for chunk in evidence
            if chunk.metadata.get(STATE_CHANGING)
            and isinstance(sequence := chunk.metadata.get(OBSERVATION_SEQUENCE), int)
        ]

        def overtaken(chunk: Chunk) -> bool:
            sequence = chunk.metadata.get(OBSERVATION_SEQUENCE)
            return (
                not chunk.metadata.get(STATE_CHANGING)
                and isinstance(sequence, int)
                and any(
                    later > sequence and named & numbers(chunk.content) for later, named in writes
                )
            )

        return [chunk for chunk in evidence if not overtaken(chunk)]

    def _is_state_record(self, chunk: Chunk) -> bool:
        """Whether this is a line of the record's state, which only the state model reads."""
        return chunk.kind in self._record_kinds and bool(self._state_line.search(chunk.content))

    async def _judge(
        self, claim: str, tier: GroundingTierSettings, chunks: list[Chunk]
    ) -> _TierOutcome:
        """One tier's answer: top-k within the tier, then NLI, latest observation first."""
        chunks = [chunk for chunk in chunks if not self._is_state_record(chunk)]
        if not chunks:
            return _TierOutcome(tier, "abstain", None, [], [], note="no evidence")
        self._retriever.index(chunks)
        retrieved = await self._retriever.retrieve(claim, self._top_k)
        relevant = [chunk for chunk in retrieved if chunk.score >= self._floor]
        restated = self._restated(claim, relevant) if "rule" in tier.claims else None
        results = (
            []
            if restated
            else list(
                await asyncio.gather(
                    *(self._entailment.entails(c.content, claim) for c in relevant)
                )
            )
        )
        judged = [(restated[0], restated[1])] if restated else list(zip(results, relevant))
        chosen = self._deciding(claim, tier, judged)
        checks = self._checks(retrieved, judged)
        label = chosen[0].label if chosen else "abstain"
        note = "" if chosen else self._why_undecided(tier, retrieved, judged)
        return _TierOutcome(tier, label, chosen[0] if chosen else None, checks, retrieved, note)

    def _why_undecided(
        self,
        tier: GroundingTierSettings,
        retrieved: list[Chunk],
        judged: list[tuple[EntailmentResult, Chunk]],
    ) -> str:
        """In a few words, why this tier could not decide the claim."""
        if not judged:
            best = max((chunk.score for chunk in retrieved), default=0.0)
            return f"nothing related (best {best:.2f} < {self._floor})"
        result = max((item[0] for item in judged), key=lambda item: item.score)
        refutation = result.label == "contradiction" and not tier.may_refute
        if refutation:
            return f"contradiction ignored, this tier cannot refute ({result.score:.2f})"
        label = _LABEL_NAMES.get(result.label, result.label)
        gloss = " (it neither states nor contradicts this)" if label == "neutral" else ""
        return f"{label} {result.score:.2f}{gloss}"

    def _deciding(
        self,
        claim: str,
        tier: GroundingTierSettings,
        judged: list[tuple[EntailmentResult, Chunk]],
    ) -> tuple[EntailmentResult, Chunk] | None:
        """The one decisive result that speaks for the tier: the latest observation, then the closest.

        Two reads of one order, before and after it was cancelled, disagree
        about "your order has been cancelled" — and the later is the world.
        Within one observation the sentence most like the claim decides,
        so a sibling sentence about something else can neither erase a
        true contradiction nor manufacture one.
        """
        decisive = [
            item
            for item in judged
            if item[0].label in _DECIDING_LABELS
            and self._is_decisive(item[0])
            and (
                item[0].label == "entailment"
                or (tier.may_refute and self._may_refute(claim, item[1]))
            )
        ]
        return max(self._one_voice_per_observation(decisive), key=self._recency, default=None)

    @staticmethod
    def _one_voice_per_observation(
        decisive: list[tuple[EntailmentResult, Chunk]],
    ) -> list[tuple[EntailmentResult, Chunk]]:
        """An observation is one consistent record, so any sentence that entails it supports it.

        Its sentences cannot really disagree; when a model says they do, the
        contradiction is the model's, about a sentence that is not about
        what the claim is about. So within one observation an entailing
        sentence outvotes a contradicting sibling, whichever sentence is
        closer to the claim. A contradiction stands only when nothing in its
        observation entails the claim. Evidence with no observation (a rule,
        a passage) is its own voice.
        """
        voices: dict[object, tuple[EntailmentResult, Chunk]] = {}
        for index, item in enumerate(decisive):
            sequence = item[1].metadata.get(OBSERVATION_SEQUENCE)
            key: object = sequence if isinstance(sequence, int) else ("alone", index)
            held = voices.get(key)
            if held is None or _outvotes(item, held):
                voices[key] = item
        return list(voices.values())

    @staticmethod
    def _recency(item: tuple[EntailmentResult, Chunk]) -> tuple[int, float]:
        sequence = item[1].metadata.get(OBSERVATION_SEQUENCE)
        return (sequence if isinstance(sequence, int) else -1, item[1].score)

    def _restated(self, claim: str, chunks: list[Chunk]) -> tuple[EntailmentResult, Chunk] | None:
        """A rule that says the claim in nearly its own words.

        "Tell the customer … they'll hear back soon" is an instruction, and
        NLI on an instruction against the statement it dictates is a coin
        toss that has come up "contradiction". The rule *is* the
        grounding for the sentence it scripts, so a rule that covers the
        claim's content, with the same polarity, supports it directly.
        """
        wanted = set(tokenize(claim)) - set(self._negations)
        for chunk in chunks:
            have = set(tokenize(chunk.content))
            coverage = len(wanted & have) / len(wanted) if wanted else 0.0
            if coverage >= self._restatement and self._negated(claim) == self._negated(
                chunk.content
            ):
                return (
                    EntailmentResult(
                        label="entailment",
                        entailed=True,
                        score=coverage,
                        calibrated=False,
                        model=RESTATEMENT_MODEL,
                        rationale=f"a rule restates {coverage:.0%} of the claim",
                    ),
                    chunk,
                )
        return None

    def _negated(self, text: str) -> bool:
        lowered = text.casefold()
        return any(re.search(rf"\b{re.escape(marker)}\b", lowered) for marker in self._negations)

    def _may_refute(self, claim: str, chunk: Chunk) -> bool:
        """Whether this item is about what the claim is about, so it may contradict it.

        A claim naming order 10876 is not refuted by the record of order
        10854 — NLI calls that a contradiction because the numbers differ.
        """
        if self._identifier is None:
            return True
        named = set(self._identifier.findall(claim))
        return not named or bool(named & set(self._identifier.findall(chunk.content)))

    def _is_decisive(self, result: EntailmentResult) -> bool:
        """Whether a result settles the claim, judged on its own scale.

        `nli_entailment_threshold` is a probability, and only a calibrated
        classifier produces one. The lexical fallback scores token
        coverage instead, and a claim that restates its evidence exactly
        still covers only the words the evidence happens to contain. So
        the threshold applies only where it means something: an
        uncalibrated model's label is its decision.
        """
        return result.score >= self._threshold if result.calibrated else True

    def _checks(
        self, retrieved: list[Chunk], judged: list[tuple[EntailmentResult, Chunk]]
    ) -> list[EvidenceCheck]:
        """Every retrieved item in similarity order, with the NLI result it got (if any)."""
        by_chunk = {id(chunk): result for result, chunk in judged}
        return [
            EvidenceCheck(
                rank=0,
                similarity=round(chunk.score, 3),
                tier=chunk.kind,
                excerpt=chunk.content,
                nli_label=by_chunk[id(chunk)].label
                if id(chunk) in by_chunk
                else f"not checked (below {self._floor})",
                nli_score=round(by_chunk[id(chunk)].score, 3) if id(chunk) in by_chunk else 0.0,
                decisive=id(chunk) in by_chunk
                and by_chunk[id(chunk)].label in _DECIDING_LABELS
                and self._is_decisive(by_chunk[id(chunk)]),
            )
            for chunk in retrieved
        ]

    @staticmethod
    def _ranked(checks: list[EvidenceCheck]) -> list[EvidenceCheck]:
        return [
            check.model_copy(update={"rank": rank}) for rank, check in enumerate(checks, start=1)
        ]

    def _verdict(
        self,
        claim: str,
        kind: ClaimClass,
        tier: str,
        result: EntailmentResult | None,
        retrieved: list[Chunk],
        *,
        supported: bool = False,
        undetermined: bool = False,
        rationale: str = "",
        reason_code: str = "",
        checks: list[EvidenceCheck] | None = None,
    ) -> DetectionVerdict:
        """An auditable verdict: who decided, on what, and everything retrieved."""
        return DetectionVerdict(
            claim=claim,
            claim_class=kind,
            supported=supported,
            undetermined=undetermined,
            confidence=result.score if result else 0.0,
            tier=tier,
            rationale=rationale or (result.rationale if result else ""),
            reason_code=reason_code,
            detector_model=(result.model if result else "")
            or (DIRECT_TIER if tier == DIRECT_TIER else self._detector_model),
            retrieval_top_k=self._top_k,
            retrieval_method=self._retrieval_method,
            retrieved_evidence=[chunk.content for chunk in retrieved],
            evidence_checks=checks or [],
            threshold=self._threshold if result is not None and result.calibrated else None,
        )

    @property
    def _retrieval_method(self) -> str:
        """Report the actual retrieval path used by the injected ranker."""
        cascade = getattr(self._retriever, "_ranker", None)
        enabled = getattr(get_settings().evals.cascade, "embedding_similarity", False)
        return "embedding" if enabled and cascade is not None else "lexical_fallback"

    @property
    def _detector_model(self) -> str:
        """Expose the concrete NLI model, including its deterministic fallback."""
        return str(getattr(self._entailment, "model_name", "")) or "lexical-entailment-fallback"
