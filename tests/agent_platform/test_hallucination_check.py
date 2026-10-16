"""Detection, diagnosis and trust calibration (LLD §10)."""

import asyncio
from typing import Any

import pytest

from agent_platform.context import Chunk
from agent_platform.guardrails.response import ResponseGuardrail
from agent_platform.hallucination_check import (
    ClaimDecomposer,
    DetectionLadder,
    GroundingReasoningDiagnoser,
    TrustCalibrationTracker,
)
from agent_platform.scoring import NOLLM_TIERS, Claim, create_scoring_cascade
from agent_platform.scoring.entailment import EntailmentResult, LexicalEntailmentModel

_EVIDENCE = [
    Chunk(
        content="Order 10001 was delivered on March 3 and refunded $42.00.",
        kind="tool_result",
        source="db",
    )
]


class TestClaimDecomposition:
    def test_splits_on_sentence_boundaries(self) -> None:
        claims = ClaimDecomposer().decompose("The order shipped. The refund was issued.")
        assert len(claims) == 2

    def test_splits_a_joined_sentence_into_separate_claims(self) -> None:
        """Three right sentences and one fabrication average out to 'mostly right'."""
        claims = ClaimDecomposer().decompose("The order shipped and the refund was issued")
        assert len(claims) == 2

    def test_a_conjunction_inside_a_condition_does_not_split(self) -> None:
        """A condition ("once approved and issued, …") is one promise, not an outcome."""
        claims = ClaimDecomposer().decompose(
            "Once approved and issued, it typically takes 5-10 business days to show on your card."
        )
        assert [claim.text for claim in claims] == [
            "Once approved and issued, it typically takes 5-10 business days to show on your card"
        ]

    def test_a_conjunction_after_the_conditions_comma_still_splits(self) -> None:
        claims = ClaimDecomposer().decompose(
            "When it settles, the refund is final and the customer gets a confirmation"
        )
        assert len(claims) == 2

    def test_a_clause_with_no_condition_still_splits_at_and(self) -> None:
        claims = ClaimDecomposer().decompose("Order 10988 has been cancelled and refunded in full")
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

    async def test_withholding_the_detector_verifies_nothing(self) -> None:
        """Off means the detector was never built, not that it runs inert.

        A ladder that exists, assesses — so the switch lives in the
        composition root, and `hallucination_guard: false` reaches here
        as a guardrail with no detector to consult.
        """
        from agent_platform.guardrails.response import ResponseGuardrail

        result = await ResponseGuardrail(grounding=None).enforce("I refunded $500", _EVIDENCE)
        assert result.grounded and not result.escalated

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


class TestSchemaValidationIsIndependentOfGrounding:
    """The coupling this replaced, asserted from both sides.

    `ResponseGuardrail` used to gate everything on
    `guardrails.enabled or hallucination_check.enabled`, so
    switching either off quietly took half of the other with it — a
    malformed envelope reaching the customer because grounding was off,
    or claims going unverified because schema checking was. They are
    different guarantees and they now have different switches.
    """

    def _guardrail(self, grounding: object) -> ResponseGuardrail:
        return ResponseGuardrail(grounding=grounding)  # type: ignore[arg-type]

    def test_the_envelope_is_still_validated_without_a_detector(self) -> None:
        reply, fields = self._guardrail(None).validate_schema('{"response": "All done."}')
        assert reply == "All done."
        assert fields["schema_valid"] is True

    def test_a_malformed_envelope_is_still_caught_without_a_detector(self) -> None:
        reply, fields = self._guardrail(None).validate_schema("not json at all")
        assert fields["schema_valid"] is False
        assert reply == "not json at all"

    async def test_a_business_claim_without_evidence_is_refused_when_grounding_is_on(
        self,
    ) -> None:
        """The deterministic rule that makes this feature demonstrable live.

        It fires on the *absence of a successful tool result*, not on the
        model misbehaving — so a refund whose tool failed reliably
        triggers it against a real model.
        """
        guardrail = ResponseGuardrail(grounding=DetectionLadder(create_scoring_cascade()))
        result = await guardrail.enforce("I've processed your refund of $500.", [])
        assert result.escalated
        assert result.reason_code == "BUSINESS_CLAIM_WITHOUT_TOOL_EVIDENCE"

    async def test_the_same_claim_reaches_the_customer_when_grounding_is_off(self) -> None:
        """And this is what the episode shows: the money never moved."""
        result = await self._guardrail(None).enforce("I've processed your refund of $500.", [])
        assert not result.escalated
        assert result.response == "I've processed your refund of $500."


class TestAThresholdIsJudgedOnItsOwnScale:
    """The bug this guards: a true claim blocked as a hallucination.

    Reported from a live run. The agent said "A refund has already been
    approved for order 10864" with "Order 10864 refund status: approved"
    in evidence — correct, and directly supported. It was escalated to a
    human with `confidence = 0.444` against `nli_entailment_threshold =
    0.600`.

    Nothing had gone wrong with the model, because no model ran. The
    hosted cross-encoder was unreachable and the lexical fallback
    answered instead, and its score is *token coverage*, not a
    probability. The sentence covers four of its nine tokens — the other
    five are function words a terse database line will never contain.
    Comparing that ratio with a probability threshold rejects perfect
    paraphrases, and the trace then blamed "NLI".
    """

    # `approved` is no longer a refund status (a reviewer's approval issues the refund),
    # so the record now says `issued`; "approved" is the same event on the ladder.
    EVIDENCE = [Chunk(content="Order 10864 refund status: issued", kind="tool_result")]

    async def _assess(self, claim: str) -> Any:
        return await DetectionLadder(create_scoring_cascade()).assess(claim, self.EVIDENCE)

    async def test_a_true_paraphrase_is_supported(self) -> None:
        """The exact claim from the reported run."""
        assessment = await self._assess("A refund has already been approved for order 10864.")
        assert assessment.clean
        assert assessment.verdicts[0].supported

    async def test_a_status_the_record_states_is_decided_by_the_record(self) -> None:
        """Saying "NLI" when the record decided sends a reader after the wrong bug."""
        assessment = await self._assess("A refund has already been approved for order 10864.")
        assert assessment.verdicts[0].detector_model == "deterministic"

    async def test_the_verdict_names_the_detector_that_decided(self) -> None:
        """Saying "NLI" when the fallback ran sends a reader after the wrong bug."""
        prose = [Chunk(content="Refunded $4.20 for order 10864", kind="tool_result")]
        assessment = await DetectionLadder(create_scoring_cascade()).assess(
            "A refund of $4.20 was issued for order 10864.", prose
        )
        assert assessment.verdicts[0].detector_model == "lexical-entailment-fallback"

    @pytest.mark.parametrize(
        "claim",
        [
            "No refund has been approved for order 10864.",
            "Order 10864 refund status: denied",
            "I have refunded $4,200 to your card for order 99999.",
            "Your package will arrive on Tuesday by courier.",
        ],
    )
    async def test_it_still_blocks_what_the_evidence_does_not_support(self, claim: str) -> None:
        """Loosening the wrong threshold must not loosen the guard itself.

        Negation, a contradicted status, a fabricated amount and an
        unrelated assertion all still fail — the fallback gates on its own
        `min_overlap` and on negation polarity, which is the check that
        actually applies to a coverage score.
        """
        assert not (await self._assess(claim)).clean

    def test_an_uncalibrated_score_says_so(self) -> None:
        """The flag that makes the distinction available to callers."""
        assert EntailmentResult(label="entailment", entailed=True, score=0.44).calibrated
        assert not asyncio.run(
            LexicalEntailmentModel().entails("Order 1 refund status: approved", "refund approved")
        ).calibrated

    async def test_the_cascade_resolves_a_paraphrase_without_a_judge(self) -> None:
        """`nollm_resolved_ratio` was 0.0 for the same reason.

        The eval tier declined every uncalibrated score and pushed the
        claim down to the LLM judge — paying for a model call to settle
        something two deterministic tiers already had.
        """
        verdict = await create_scoring_cascade().resolve(
            Claim(
                text="A refund has already been approved for order 10864",
                evidence=["Order 10864 refund status: approved"],
            )
        )
        assert verdict.label == "supported"
        assert verdict.tier in NOLLM_TIERS


@pytest.mark.usefixtures("abstaining_guard")
class TestAnAbstentionIsNotAHallucination:
    """Found in a live run: a true, tool-grounded reply was blocked.

    The customer asked about a refund; the agent answered "your refund
    request has not been initiated; the refund status currently shows
    none", against a tool result reading `Order 10979 refund status:
    none`. Every word of it was supported. The customer was shown "I
    couldn't verify all of the details in that response."

    The lexical fallback had returned `label="unknown"` — token coverage
    36%, below its own 40% floor — which is the detector saying it cannot
    tell. That reached the guardrail as `supported=False` and blocked the
    reply. A heuristic that abstains must not be able to silence a
    correct answer; only evidence to the contrary should.
    """

    EVIDENCE = [
        Chunk(content="Order 10979 refund status: none", kind="tool_result"),
        Chunk(
            content=(
                "Order 10979: Laptop Stand, Aluminium, delivered, "
                "delivered on 2026-09-24, total $56.86"
            ),
            kind="tool_result",
        ),
    ]
    DRAFT = (
        "Your refund request for order 10979 has not been initiated; "
        "the refund status currently shows none."
    )

    def test_the_live_draft_is_not_blocked(self) -> None:
        guard = ResponseGuardrail(DetectionLadder(create_scoring_cascade()))
        result = asyncio.run(guard.enforce(self.DRAFT, self.EVIDENCE))
        assert result.escalated is False
        assert result.response == self.DRAFT

    def test_an_abstention_is_recorded_rather_than_counted_against(self) -> None:
        """The trace still shows what went unchecked."""
        assessment = asyncio.run(
            DetectionLadder(create_scoring_cascade()).assess(self.DRAFT, self.EVIDENCE)
        )
        assert assessment.clean is True
        assert assessment.unsupported == []
        assert assessment.undetermined

    def test_a_claim_the_record_refutes_is_still_blocked(self) -> None:
        """The guard must still do the job it exists for."""
        guard = ResponseGuardrail(DetectionLadder(create_scoring_cascade()))
        result = asyncio.run(
            guard.enforce(
                "The refund for order 10979 has been completed and the money was sent.",
                self.EVIDENCE,
            )
        )
        assert result.escalated is True
        assert result.reason_code == "TOOL_EVIDENCE_RECORD_STATE_MISMATCH"

    def test_a_negated_claim_is_not_read_as_an_assertion(self) -> None:
        """ "No refund has been processed" carries the settled wording and denies it."""
        guard = ResponseGuardrail(DetectionLadder(create_scoring_cascade()))
        result = asyncio.run(
            guard.enforce("No refund has been processed for order 10979 yet.", self.EVIDENCE)
        )
        assert result.escalated is False

    def test_a_settled_claim_matching_the_record_passes(self) -> None:
        settled = [Chunk(content="Order 10979 refund status: completed", kind="tool_result")]
        guard = ResponseGuardrail(DetectionLadder(create_scoring_cascade()))
        result = asyncio.run(
            guard.enforce("The refund for order 10979 has been completed.", settled)
        )
        assert result.escalated is False


class TestClaimsSplitOnSemicolons:
    """The draft above was one claim because `;` was not a boundary.

    Its two halves were then scored together against evidence that
    matched only one of them, so a sentence half of which was verbatim
    supported scored as entirely unsupported.
    """

    def test_a_semicolon_separates_two_claims(self) -> None:
        claims = ClaimDecomposer().decompose(
            "Your refund request has not been initiated; the refund status shows none."
        )
        assert [claim.text for claim in claims] == [
            "Your refund request has not been initiated",
            "the refund status shows none",
        ]


class TestRepairDoesNotRunTwiceForNothing:
    """The reviser returning the draft unchanged means it had nothing to remove."""

    def test_an_unchanged_revision_ends_the_loop(self) -> None:
        calls: list[str] = []

        async def revise(candidate: str, unsupported: list[str]) -> str:
            calls.append(candidate)
            return candidate

        guard = ResponseGuardrail(DetectionLadder(create_scoring_cascade()))
        evidence = [Chunk(content="Order 10001 refund status: none", kind="tool_result")]
        asyncio.run(
            guard.enforce("The refund for order 10001 has been completed.", evidence, revise=revise)
        )
        # The deterministic refund-state check settles this before any
        # repair is attempted, so the model is never asked.
        assert calls == []
