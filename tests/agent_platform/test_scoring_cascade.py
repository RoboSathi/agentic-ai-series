"""The NOLLM scoring cascade: claims resolved cheapest tier first (LLD §18)."""

from agent_platform.llm.embeddings import EmbeddingPurpose, HashingEmbeddingClient
from agent_platform.scoring import (
    Claim,
    DeterministicRuleScorer,
    ExactAssertionScorer,
    HumanAdjudicationScorer,
    LexicalEntailmentModel,
    NliEntailmentScorer,
    NollmScoringCascadeImpl,
    Verdict,
)
from agent_platform.scoring.cascade import EmbeddingSimilarityScorer


class TestExactAssertion:
    async def test_matches_an_expected_value(self) -> None:
        verdict = await ExactAssertionScorer().resolve(Claim(text="42", expected="42"))
        assert verdict is not None and verdict.label == "supported"

    async def test_reports_a_mismatch(self) -> None:
        verdict = await ExactAssertionScorer().resolve(Claim(text="41", expected="42"))
        assert verdict is not None and verdict.label == "contradicted"

    async def test_finds_a_claim_verbatim_in_evidence(self) -> None:
        verdict = await ExactAssertionScorer().resolve(
            Claim(text="refund approved", evidence=["the refund approved yesterday"])
        )
        assert verdict is not None and verdict.label == "supported"

    async def test_declines_when_nothing_is_literal(self) -> None:
        """Declining is the correct answer — the next tier will try."""
        assert await ExactAssertionScorer().resolve(Claim(text="x", evidence=["y"])) is None


class TestDeterministicRules:
    async def test_a_registered_rule_resolves(self) -> None:
        scorer = DeterministicRuleScorer()
        scorer.register(
            "under_limit",
            lambda claim: Verdict(label="supported", score=1.0, tier="deterministic_rule"),
        )
        verdict = await scorer.resolve(Claim(text="x"))
        assert verdict is not None and verdict.label == "supported"

    async def test_declines_when_no_rule_applies(self) -> None:
        scorer = DeterministicRuleScorer()
        scorer.register("never", lambda claim: None)
        assert await scorer.resolve(Claim(text="x")) is None


class TestEntailment:
    """The fallback model, for when no hosted cross-encoder is reachable."""

    async def test_coverage_plus_matching_polarity_entails(self) -> None:
        result = await LexicalEntailmentModel().entails(
            "the refund was approved today", "the refund was approved"
        )
        assert result.entailed is True

    async def test_opposing_polarity_does_not_entail(self) -> None:
        """The failure similarity alone cannot catch."""
        result = await LexicalEntailmentModel().entails(
            "the refund was not approved", "the refund was approved"
        )
        assert result.entailed is False

    async def test_low_coverage_scores_zero(self) -> None:
        scored = await LexicalEntailmentModel().entails("weather is nice", "refund approved")
        assert scored.score == 0.0

    async def test_the_scorer_declines_without_evidence(self) -> None:
        assert await NliEntailmentScorer().resolve(Claim(text="x")) is None


class TestHumanAdjudication:
    async def test_defers_rather_than_fabricating(self) -> None:
        verdict = await HumanAdjudicationScorer().resolve(Claim(text="unknowable"))
        assert verdict is not None and verdict.label == "deferred"


class TestCascade:
    async def test_resolves_at_the_cheapest_sufficient_tier(self) -> None:
        cascade = NollmScoringCascadeImpl()
        verdict = await cascade.resolve(Claim(text="42", expected="42"))
        assert verdict.tier == "exact_assertion"

    async def test_falls_through_to_entailment(self) -> None:
        cascade = NollmScoringCascadeImpl()
        verdict = await cascade.resolve(
            Claim(
                text="the refund was approved", evidence=["we approved the refund for this order"]
            )
        )
        assert verdict.tier in {"nli_entailment", "embedding_similarity"}

    async def test_a_negated_claim_is_not_silently_supported(self) -> None:
        """Similarity would call these a match; the cascade must not."""
        cascade = NollmScoringCascadeImpl()
        verdict = await cascade.resolve(
            Claim(text="the refund was approved", evidence=["the refund was denied"])
        )
        assert verdict.label != "supported"

    async def test_reports_its_own_nollm_ratio(self) -> None:
        cascade = NollmScoringCascadeImpl()
        await cascade.resolve(Claim(text="42", expected="42"))
        assert cascade.nollm_scorer_ratio == 1.0

    async def test_ratio_is_one_before_anything_is_resolved(self) -> None:
        assert NollmScoringCascadeImpl().nollm_scorer_ratio == 1.0

    async def test_tier_counts_are_tracked(self) -> None:
        cascade = NollmScoringCascadeImpl()
        await cascade.resolve(Claim(text="42", expected="42"))
        assert cascade.tier_counts["exact_assertion"] == 1

    async def test_a_disabled_tier_is_skipped(self, restore_settings: None) -> None:
        from agent_platform.config import get_settings

        get_settings().hallucination_check.scoring.tiers.exact_assertion = False
        cascade = NollmScoringCascadeImpl()
        verdict = await cascade.resolve(Claim(text="42", expected="42"))
        assert verdict.tier != "exact_assertion"


class TestSimilarityPurposes:
    """An asymmetric model needs the question as a query and the candidates as passages."""

    async def test_the_text_is_a_query_and_the_candidates_are_passages(self) -> None:
        asked: list[tuple[str, list[str]]] = []

        class _Recording(HashingEmbeddingClient):
            async def embed(
                self, texts: list[str], purpose: EmbeddingPurpose = "query"
            ) -> list[list[float]]:
                asked.append((purpose, texts))
                return await super().embed(texts, purpose)

        scores = await EmbeddingSimilarityScorer(_Recording()).similarities(
            "refund status", ["refund issued", "shipping times"]
        )
        assert len(scores) == 2 and scores[0] > scores[1]
        assert asked == [
            ("query", ["refund status"]),
            ("passage", ["refund issued", "shipping times"]),
        ]
