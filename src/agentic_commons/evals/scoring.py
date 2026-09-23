"""The NOLLM Scoring Cascade (E10, LLD §18).

One implementation, six tiers, four consumers. `context` retrieval
tiering, `tools` selection, `llm` model cascading and
`hallucination_protection` detection all resolve claims *here* rather
than each growing its own scorer — which is the only way
`nollm_scorer_ratio` means anything across the system.

The ordering is the whole design: every tier is cheaper, faster and more
deterministic than the one below it, so a claim only falls through to a
model when nothing decisive is available above. Reaching the LLM judge is
a failure to resolve cheaply, not a first move.
"""

from typing import Callable, Protocol, runtime_checkable

from agentic_commons.config import get_settings
from agentic_commons.evals.entailment import EntailmentModel, create_entailment_model
from agentic_commons.evals.judge import CalibratedJudge
from agentic_commons.evals.types import Claim, ScorerTier, Verdict
from agentic_commons.llm.embeddings import (
    EmbeddingClient,
    cosine_similarity,
    create_embedding_client,
    tokenize,
)
from agentic_commons.logging import get_logger, instrument

# Cascade order. Everything above the judge is a NOLLM tier by definition.
TIER_ORDER: tuple[ScorerTier, ...] = (
    "exact_assertion",
    "deterministic_rule",
    "embedding_similarity",
    "nli_entailment",
    "calibrated_llm_judge",
    "human_adjudication",
)

_JUDGE_INDEX = TIER_ORDER.index("calibrated_llm_judge")
NOLLM_TIERS: frozenset[ScorerTier] = frozenset(TIER_ORDER[:_JUDGE_INDEX])

DeterministicRule = Callable[[Claim], Verdict | None]


@runtime_checkable
class Scorer(Protocol):
    """Resolves a claim, or declines so the next tier may try."""

    tier: ScorerTier

    async def resolve(self, claim: Claim) -> Verdict | None: ...


@runtime_checkable
class NollmScoringCascade(Protocol):
    """Resolves claims through the tiered cascade and reports its own mix."""

    async def resolve(self, claim: Claim) -> Verdict: ...

    @property
    def nollm_scorer_ratio(self) -> float: ...


class ExactAssertionScorer:
    """Tier 1 — literal equality, or the claim appearing verbatim in evidence."""

    tier: ScorerTier = "exact_assertion"

    async def resolve(self, claim: Claim) -> Verdict | None:
        """Resolve only when equality is unambiguous; otherwise decline."""
        if claim.expected is not None:
            matched = str(claim.expected).strip() == claim.text.strip()
            return self._verdict(matched, "expected value compared literally")
        needle = claim.text.strip().casefold()
        if needle and any(needle in item.casefold() for item in claim.evidence):
            return self._verdict(True, "claim found verbatim in evidence")
        return None

    def _verdict(self, matched: bool, rationale: str) -> Verdict:
        return Verdict(
            label="supported" if matched else "contradicted",
            score=1.0 if matched else 0.0,
            tier=self.tier,
            rationale=rationale,
        )


class DeterministicRuleScorer:
    """Tier 2 — registered domain predicates (Strategy + Registry).

    Rules are how a subsystem injects the parts of its judgement that are
    genuinely deterministic — a refund is under the threshold, a tool's
    namespace matches the task — without the cascade learning that domain.
    """

    tier: ScorerTier = "deterministic_rule"

    def __init__(self) -> None:
        self._rules: dict[str, DeterministicRule] = {}

    def register(self, name: str, rule: DeterministicRule) -> None:
        """Add a named rule. Rules run in registration order."""
        self._rules[name] = rule

    async def resolve(self, claim: Claim) -> Verdict | None:
        """First rule that returns a verdict wins."""
        for name, rule in self._rules.items():
            verdict = rule(claim)
            if verdict is not None:
                return verdict.model_copy(
                    update={"tier": self.tier, "rationale": verdict.rationale or f"rule:{name}"}
                )
        return None


class EmbeddingSimilarityScorer:
    """Tier 3 — cosine similarity against the strongest piece of evidence.

    Resolves only at the confident ends of the range: clearly similar, or
    clearly unrelated. The ambiguous middle is exactly what the tiers
    below exist for, so it declines there rather than guessing.
    """

    tier: ScorerTier = "embedding_similarity"

    def __init__(self, embeddings: EmbeddingClient | None = None) -> None:
        settings = get_settings().evals
        self._embeddings = embeddings or create_embedding_client()
        self._similar = settings.embedding_similarity_threshold
        self._dissimilar = settings.embedding_dissimilarity_threshold

    async def resolve(self, claim: Claim) -> Verdict | None:
        """Resolve at the confident ends of the similarity range only."""
        if not claim.evidence:
            return None
        best = max(await self.similarities(claim.text, claim.evidence), default=0.0)
        if best >= self._similar:
            return self._verdict("supported", best)
        if best <= self._dissimilar:
            return self._verdict("contradicted", best)
        return None

    async def similarities(self, text: str, candidates: list[str]) -> list[float]:
        """Cosine similarity of `text` against each candidate, one batch call."""
        if not candidates:
            return []
        vectors = await self._embeddings.embed([text, *candidates])
        query, rest = vectors[0], vectors[1:]
        return [cosine_similarity(query, vector) for vector in rest]

    def _verdict(self, label: str, score: float) -> Verdict:
        return Verdict(
            label="supported" if label == "supported" else "contradicted",
            score=score,
            tier=self.tier,
            rationale=f"cosine similarity {score:.3f}",
        )


class NliEntailmentScorer:
    """Tier 4 — does the evidence *entail* the claim, not merely resemble it.

    Similarity and entailment part company exactly where it matters: "the
    refund was approved" and "the refund was denied" are near-identical
    strings and opposite facts.
    """

    tier: ScorerTier = "nli_entailment"

    def __init__(self, model: EntailmentModel | None = None) -> None:
        self._model = model or create_entailment_model()
        self._threshold = get_settings().evals.nli_entailment_threshold

    async def resolve(self, claim: Claim) -> Verdict | None:
        """Resolve when entailment or contradiction is confident enough."""
        if not claim.evidence:
            return None
        result = await self._model.entails(" ".join(claim.evidence), claim.text)
        # The threshold is a probability, so it applies only to a model that
        # returns one. Against the lexical fallback's token coverage it
        # declined every paraphrase and pushed it down to the judge, which
        # is why nollm_resolved_ratio sat at zero.
        if result.calibrated and result.score < self._threshold:
            return None
        if result.label == "unknown":
            return None
        return Verdict(
            label="supported" if result.entailed else "contradicted",
            score=result.score,
            tier=self.tier,
            rationale=result.rationale,
        )


class CalibratedJudgeScorer:
    """Tier 5 — an LLM judge, trusted only above its calibrated confidence."""

    tier: ScorerTier = "calibrated_llm_judge"

    def __init__(self, judge: CalibratedJudge | None = None) -> None:
        self._judge = judge
        self._threshold = get_settings().evals.judge_confidence_threshold

    async def resolve(self, claim: Claim) -> Verdict | None:
        """Decline when no judge is wired, or when it is not confident."""
        if self._judge is None:
            return None
        verdict = await self._judge.judge(claim)
        if verdict.score < self._threshold:
            return None
        return verdict.model_copy(update={"tier": self.tier})


class HumanAdjudicationScorer:
    """Tier 6 — nothing resolved it, so a person must.

    This tier never fabricates an answer. It returns `deferred`, which is
    an honest verdict and a queue entry, not a scoring failure.
    """

    tier: ScorerTier = "human_adjudication"

    async def resolve(self, claim: Claim) -> Verdict | None:
        """Always defers — the terminal tier of the cascade."""
        return Verdict(
            label="deferred",
            score=0.0,
            tier=self.tier,
            rationale="no automated tier resolved this claim",
        )


@instrument("evals")
class NollmScoringCascadeImpl:
    """Chain of Responsibility over the six tiers.

    Tiers are assembled from config, so an episode can switch one off and
    watch the ratio, the cost and the latency move together (E10).
    """

    def __init__(
        self,
        embeddings: EmbeddingClient | None = None,
        entailment: EntailmentModel | None = None,
        judge: CalibratedJudge | None = None,
        rules: DeterministicRuleScorer | None = None,
    ) -> None:
        toggles = get_settings().evals.cascade
        self.rules = rules or DeterministicRuleScorer()
        self._similarity = EmbeddingSimilarityScorer(embeddings)
        candidates: dict[ScorerTier, Scorer] = {
            "exact_assertion": ExactAssertionScorer(),
            "deterministic_rule": self.rules,
            "embedding_similarity": self._similarity,
            "nli_entailment": NliEntailmentScorer(entailment),
            "calibrated_llm_judge": CalibratedJudgeScorer(judge),
            "human_adjudication": HumanAdjudicationScorer(),
        }
        self._tiers = [candidates[tier] for tier in TIER_ORDER if getattr(toggles, tier)]
        self._resolutions: dict[ScorerTier, int] = {}
        self._logger = get_logger("evals")

    async def resolve(self, claim: Claim) -> Verdict:
        """Walk the cascade top-down, returning the first decisive verdict."""
        for scorer in self._tiers:
            verdict = await scorer.resolve(claim)
            if verdict is not None:
                self._record(verdict.tier)
                return verdict
        return self._unresolved()

    async def rank(self, text: str, candidates: list[str]) -> list[float]:
        """Similarity scores for a ranking consumer (tool and context retrieval).

        Ranking is not claim resolution: callers that need an ordering, not
        a verdict, get the similarity tier directly rather than abusing
        `resolve()` for a number it deliberately refuses to produce in the
        ambiguous middle.
        """
        if get_settings().evals.cascade.embedding_similarity:
            return await self._similarity.similarities(text, candidates)
        return [lexical_overlap(text, candidate) for candidate in candidates]

    @property
    def nollm_scorer_ratio(self) -> float:
        """Share of resolutions settled without an LLM (E10's headline metric)."""
        total = sum(self._resolutions.values())
        if not total:
            return 1.0
        nollm = sum(count for tier, count in self._resolutions.items() if tier in NOLLM_TIERS)
        return nollm / total

    @property
    def tier_counts(self) -> dict[ScorerTier, int]:
        """Resolutions per tier — the cascade's own telemetry."""
        return dict(self._resolutions)

    def _record(self, tier: ScorerTier) -> None:
        self._resolutions[tier] = self._resolutions.get(tier, 0) + 1

    def _unresolved(self) -> Verdict:
        """Every tier declined or was disabled — say so rather than guess."""
        self._logger.debug("Cascade exhausted with every tier declining")
        return Verdict(
            label="unresolved",
            score=0.0,
            tier=TIER_ORDER[-1],
            rationale="every configured tier declined to resolve",
        )


def lexical_overlap(left: str, right: str) -> float:
    """Jaccard token overlap — the last-resort ranking when embeddings are off."""
    left_tokens, right_tokens = set(tokenize(left)), set(tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def create_scoring_cascade(
    embeddings: EmbeddingClient | None = None,
    entailment: EntailmentModel | None = None,
    judge: CalibratedJudge | None = None,
) -> NollmScoringCascadeImpl:
    """Build the cascade with whatever backends are available (Factory)."""
    return NollmScoringCascadeImpl(embeddings=embeddings, entailment=entailment, judge=judge)
