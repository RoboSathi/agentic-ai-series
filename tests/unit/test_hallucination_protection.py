"""Detection, diagnosis and trust calibration (LLD §10)."""

from agentic_commons.context import Chunk
from agentic_commons.evals import create_scoring_cascade
from agentic_commons.hallucination_protection import (
    ClaimDecomposer,
    DetectionLadder,
    GroundingReasoningDiagnoser,
    TrustCalibrationTracker,
)

_EVIDENCE = [
    Chunk(content="Order 10001 was delivered on March 3 and refunded $42.00.", source="db")
]


class TestClaimDecomposition:
    def test_splits_on_sentence_boundaries(self) -> None:
        claims = ClaimDecomposer().decompose("The order shipped. The refund was issued.")
        assert len(claims) == 2

    def test_splits_a_joined_sentence_into_separate_claims(self) -> None:
        """Three right sentences and one fabrication average out to 'mostly right'."""
        claims = ClaimDecomposer().decompose("The order shipped and the refund was issued")
        assert len(claims) == 2

    def test_marks_contentless_claims_unverifiable(self) -> None:
        claims = ClaimDecomposer().decompose("Sure!")
        assert claims[0].is_verifiable is False

    def test_empty_input_yields_no_claims(self) -> None:
        assert ClaimDecomposer().decompose("") == []


class TestDetectionLadder:
    async def test_a_grounded_claim_is_supported(self) -> None:
        ladder = DetectionLadder(create_scoring_cascade())
        verdict = await ladder.score("Order 10001 was delivered on March 3", _EVIDENCE)
        assert verdict.supported is True

    async def test_a_fabricated_claim_is_not_supported(self) -> None:
        ladder = DetectionLadder(create_scoring_cascade())
        verdict = await ladder.score("The customer received a $500 gift card", _EVIDENCE)
        assert verdict.supported is False

    async def test_assessing_a_generation_reports_a_rate(self) -> None:
        ladder = DetectionLadder(create_scoring_cascade())
        assessment = await ladder.assess(
            "Order 10001 was delivered on March 3. The customer received a $500 gift card.",
            _EVIDENCE,
        )
        assert 0.0 < assessment.hallucination_rate <= 1.0
        assert assessment.unsupported

    async def test_a_fully_grounded_generation_is_clean(self) -> None:
        ladder = DetectionLadder(create_scoring_cascade())
        assessment = await ladder.assess("Order 10001 was delivered on March 3", _EVIDENCE)
        assert assessment.clean is True

    async def test_the_ladder_reports_the_shared_cascades_ratio(self) -> None:
        """Not its own — one number for the whole system."""
        cascade = create_scoring_cascade()
        ladder = DetectionLadder(cascade)
        await ladder.assess("Order 10001 was delivered on March 3", _EVIDENCE)
        assert ladder is not None
        assert cascade.nollm_scorer_ratio == 1.0

    async def test_disabled_protection_assesses_nothing(self, restore_settings: None) -> None:
        from agentic_commons.config import get_settings

        get_settings().hallucination_protection.enabled = False
        ladder = DetectionLadder(create_scoring_cascade())
        assert (await ladder.assess("anything", _EVIDENCE)).verdicts == []

    def test_consistency_is_high_for_identical_samples(self) -> None:
        ladder = DetectionLadder(create_scoring_cascade())
        assert ladder.consistency(["the refund was $42", "the refund was $42"]) == 1.0

    def test_consistency_is_low_for_diverging_samples(self) -> None:
        """A fabricated detail varies between samples; a grounded one does not."""
        ladder = DetectionLadder(create_scoring_cascade())
        assert ladder.consistency(["the refund was $42", "a gift card worth 500 dollars"]) < 0.5

    def test_a_single_sample_is_trivially_consistent(self) -> None:
        assert DetectionLadder(create_scoring_cascade()).consistency(["only one"]) == 1.0


class TestDiagnosis:
    def test_missing_evidence_is_a_grounding_failure(self) -> None:
        """No amount of prompt work fixes evidence that was never retrieved."""
        assert GroundingReasoningDiagnoser().diagnose("anything", []) == "grounding"

    def test_present_evidence_makes_it_a_reasoning_failure(self) -> None:
        """More retrieval makes this one worse."""
        diagnosis = GroundingReasoningDiagnoser().diagnose(
            "Order 10001 was delivered March 3", _EVIDENCE
        )
        assert diagnosis == "reasoning"

    def test_irrelevant_evidence_is_still_a_grounding_failure(self) -> None:
        unrelated = [Chunk(content="the weather forecast for tuesday", source="x")]
        assert GroundingReasoningDiagnoser().diagnose("refund of $42", unrelated) == "grounding"


class TestTrustCalibration:
    def test_measures_automation_bias(self) -> None:
        """Accepting a wrong answer."""
        tracker = TrustCalibrationTracker()
        tracker.record(agent_correct=False, human_accepted=True)
        assert tracker.report().automation_bias_rate == 1.0

    def test_measures_algorithm_aversion(self) -> None:
        """Rejecting a right one — the opposite failure, equally expensive."""
        tracker = TrustCalibrationTracker()
        tracker.record(agent_correct=True, human_accepted=False)
        assert tracker.report().algorithm_aversion_rate == 1.0

    def test_measures_appropriate_reliance(self) -> None:
        tracker = TrustCalibrationTracker()
        tracker.record(agent_correct=True, human_accepted=True)
        tracker.record(agent_correct=False, human_accepted=False)
        assert tracker.report().appropriate_reliance_rate == 1.0

    def test_an_empty_report_is_all_zeroes(self) -> None:
        assert TrustCalibrationTracker().report().samples == 0
