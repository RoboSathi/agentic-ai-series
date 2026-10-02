"""The NOLLM cascade, judge calibration and the eval pyramid (LLD §18)."""

import pytest

from agentic_commons.evals import (
    Claim,
    ComponentEvalRunner,
    DeterministicRuleScorer,
    EvalCase,
    ExactAssertionScorer,
    GoldenDatasetProvenanceLedger,
    HumanAdjudicationScorer,
    JudgeCalibrator,
    LabelProvenance,
    LexicalEntailmentModel,
    NliEntailmentScorer,
    NollmScoringCascadeImpl,
    Verdict,
    load_manifest,
)
from agentic_commons.evals.judge import LlmJudge
from agentic_commons.persistence import SqliteConnectionPool


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
        from agentic_commons.config import get_settings

        get_settings().evals.cascade.exact_assertion = False
        cascade = NollmScoringCascadeImpl()
        verdict = await cascade.resolve(Claim(text="42", expected="42"))
        assert verdict.tier != "exact_assertion"


class TestJudge:
    def test_no_judge_is_built_when_the_tier_is_off(self, restore_settings: None) -> None:
        from agentic_commons.config import get_settings

        get_settings().evals.cascade.calibrated_llm_judge = False
        assert LlmJudge.create_if_enabled() is None

    async def test_an_unparseable_verdict_scores_zero(self, scripted_llm) -> None:
        """Which sits below the threshold, so the cascade escalates."""
        scripted_llm.script = [{"content": "I'm not sure, honestly"}]
        verdict = await LlmJudge(scripted_llm).judge(Claim(text="x"))
        assert verdict.score == 0.0

    async def test_parses_a_well_formed_verdict(self, scripted_llm) -> None:
        scripted_llm.script = [
            {"content": '{"label": "supported", "confidence": 0.9, "rationale": "clear"}'}
        ]
        verdict = await LlmJudge(scripted_llm).judge(Claim(text="x"))
        assert verdict.label == "supported" and verdict.score == 0.9

    async def test_confidence_is_clamped_to_a_probability(self, scripted_llm) -> None:
        scripted_llm.script = [
            {"content": '{"label": "supported", "confidence": 5.0, "rationale": "over"}'}
        ]
        assert (await LlmJudge(scripted_llm).judge(Claim(text="x"))).score == 1.0


class TestJudgeCalibration:
    def test_perfect_agreement_gives_kappa_one(self) -> None:
        report = JudgeCalibrator().agreement(["a", "b", "a"], ["a", "b", "a"])
        assert report.cohens_kappa == pytest.approx(1.0)

    def test_kappa_discounts_chance_agreement(self) -> None:
        """90% accuracy on a 90/10 split is worth nothing, and kappa says so."""
        judge = ["yes"] * 10
        human = ["yes"] * 9 + ["no"]
        report = JudgeCalibrator().agreement(judge, human)
        assert report.observed_agreement == 0.9
        assert report.cohens_kappa < 0.1

    def test_mismatched_lengths_are_rejected(self) -> None:
        with pytest.raises(ValueError):
            JudgeCalibrator().agreement(["a"], ["a", "b"])

    def test_an_empty_sample_is_not_acceptable(self) -> None:
        assert JudgeCalibrator().agreement([], []).acceptable is False

    def test_pairwise_win_rate(self) -> None:
        assert JudgeCalibrator.pairwise_win_rate(["a", "a", "b"], "a") == pytest.approx(2 / 3)

    def test_standard_error_shrinks_with_sample_size(self) -> None:
        calibrator = JudgeCalibrator()
        assert calibrator.standard_error(0.5, 1000) < calibrator.standard_error(0.5, 10)


class TestProvenanceLedger:
    def test_records_and_reads_back_a_label(self, pool: SqliteConnectionPool) -> None:
        ledger = GoldenDatasetProvenanceLedger(pool)
        ledger.record(
            LabelProvenance(case_id="c1", label="approve", basis="written_policy", labeller="ada")
        )
        assert ledger.get("c1") is not None

    def test_relabelling_preserves_what_it_replaced(self, pool: SqliteConnectionPool) -> None:
        """A corrected eval set whose corrections are invisible is unreviewable."""
        ledger = GoldenDatasetProvenanceLedger(pool)
        ledger.record(
            LabelProvenance(
                case_id="c1", label="deny", basis="historical_practice", labeller="legacy"
            )
        )
        updated = ledger.relabel("c1", "approve", "ada", "policy says approve")
        assert updated.superseded == "deny" and updated.basis == "written_policy"

    def test_bias_report_counts_practice_based_labels(self, pool: SqliteConnectionPool) -> None:
        ledger = GoldenDatasetProvenanceLedger(pool)
        ledger.record(
            LabelProvenance(case_id="a", label="x", basis="historical_practice", labeller="l")
        )
        ledger.record(LabelProvenance(case_id="b", label="x", basis="written_policy", labeller="l"))
        assert ledger.bias_report().practice_ratio == 0.5


class TestManifest:
    def test_the_manifest_is_checked_in_and_loads(self) -> None:
        assert load_manifest().entries

    def test_every_entry_declares_a_scorer_and_a_threshold(self) -> None:
        for entry in load_manifest().entries:
            assert entry.scorer and entry.regression_threshold > 0

    def test_reports_components_with_no_declared_eval(self) -> None:
        """The gap is meant to be visible, not implicit."""
        missing = load_manifest().missing_coverage(["evals.scoring", "not_covered"])
        assert missing == ["not_covered"]


class TestPyramid:
    async def test_runs_cases_and_aggregates(self) -> None:
        runner = ComponentEvalRunner()
        cases = [EvalCase(case_id="c1", component="x", input="q", expected="answer")]

        async def system(case: EvalCase) -> str:
            return "answer"

        report = await runner.run(cases, system)
        assert report.pass_rate == 1.0 and report.level == "component"

    async def test_a_raising_system_is_a_failed_case_not_a_crash(self) -> None:
        runner = ComponentEvalRunner()

        async def system(case: EvalCase) -> str:
            raise RuntimeError("boom")

        report = await runner.run([EvalCase(case_id="c", component="x", input="q")], system)
        assert report.pass_rate == 0.0

    async def test_an_empty_run_is_not_a_pass(self) -> None:
        async def system(case: EvalCase) -> str:
            return ""

        assert (await ComponentEvalRunner().run([], system)).pass_rate == 0.0
