"""The refund state machine, read the way NLI cannot.

`Order 10665 refund status: pending_review.` is a state record. A
cross-encoder read it as contradicting "has been submitted" (0.99), and a
true reply was blocked. The record has an explicit state machine, so it is
read by a small deterministic model of that machine, in both directions.
"""

import pytest

from agent_platform.context.types import Chunk
from agent_platform.hallucination_check.states import StateModel


def _record(*lines: str) -> list[Chunk]:
    return [
        Chunk(content=line, kind="tool_result", metadata={"observation_sequence": index})
        for index, line in enumerate(lines)
    ]


def _status(name: str) -> list[Chunk]:
    return _record(f"Order 10665 refund status: {name}.")


def _holds(status: str, claim: str) -> bool | None:
    finding = StateModel().judge(claim, _status(status))
    return None if finding is None else finding.holds


class TestAnEventIsTrueOnceTheRefundHasReachedIt:
    @pytest.mark.parametrize(
        "status, claim, expected",
        [
            ("pending_review", "Your refund request for order 10665 has been submitted", True),
            ("pending_review", "Your refund request for order 10665 has been received", True),
            ("pending_review", "The refund request was queued for human review", True),
            ("pending_review", "Your refund for order 10665 has been approved", False),
            ("pending_review", "Your refund for order 10665 has been refunded", False),
            ("pending_review", "The refund for order 10665 has been completed", False),
            ("none", "Your refund request for order 10665 has been submitted", False),
            ("none", "Your refund request for order 10665 has not been initiated", True),
            ("none", "No refund has been requested for order 10665", True),
            ("issued", "Your refund for order 10665 has been approved", True),
            ("issued", "Your refund request for order 10665 has been submitted", True),
            ("issued", "The refund for order 10665 has been completed", False),
            ("completed", "Your refund for order 10665 has been processed", True),
            ("completed", "$21.35 was returned", True),
            ("completed", "Your refund request for order 10665 has been submitted", True),
            ("completed", "The refund for order 10665 has not been refunded", False),
            ("denied", "Your refund request for order 10665 has been denied", True),
            ("denied", "Your refund request for order 10665 has been submitted", True),
            ("denied", "Your refund for order 10665 has been approved", False),
        ],
    )
    def test_the_ladder(self, status: str, claim: str, expected: bool) -> None:
        assert _holds(status, claim) is expected


class TestAStateIsTrueOnlyWhileTheRefundIsThere:
    @pytest.mark.parametrize(
        "status, claim, expected",
        [
            ("pending_review", "Your refund request for order 10665 is now pending review", True),
            ("pending_review", "Your refund request is awaiting a decision", True),
            ("completed", "Your refund request for order 10665 is pending review", False),
            ("issued", "Your refund request for order 10665 is under review", False),
            ("none", "Your refund request for order 10665 is pending review", False),
            ("denied", "Your refund request for order 10665 was denied", True),
            ("pending_review", "Your refund request for order 10665 was denied", False),
        ],
    )
    def test_exact(self, status: str, claim: str, expected: bool) -> None:
        assert _holds(status, claim) is expected

    def test_an_event_word_outranks_a_state_word_in_the_same_claim(self) -> None:
        """ "Was queued for human review" is history, and history stays true."""
        assert _holds("completed", "The refund was queued for human review") is True


class TestWhatItWillNotJudge:
    def test_no_status_in_the_record_defers(self) -> None:
        finding = StateModel().judge("Your refund has been submitted", _record("Order 10665: Lamp"))
        assert finding is None

    def test_a_claim_that_names_no_refund_state_defers(self) -> None:
        assert _holds("completed", "Your order shipped on Tuesday") is None


class TestTheCurrentStatusIsTheLatestOne:
    def test_a_later_observation_overrides_an_earlier_one(self) -> None:
        record = _record(
            "Order 10665 refund status: none.",
            "Order 10665 refund status: pending_review.",
        )
        finding = StateModel().judge("Your refund request has been submitted", record)
        assert finding is not None and finding.holds

    def test_the_status_a_referral_states_in_prose_counts(self) -> None:
        record = _record("Refund of $18.70 for order 10001 has been queued for human review")
        finding = StateModel().judge("The refund has been completed", record)
        assert finding is not None and not finding.holds

    def test_it_names_the_line_it_read(self) -> None:
        finding = StateModel().judge("Your refund has been submitted", _status("pending_review"))
        assert finding is not None
        assert finding.chunk.content == "Order 10665 refund status: pending_review."
        assert "pending_review" in finding.rationale


class TestAPromiseHoldsWhileTheRefundCanStillGetThere:
    """ "Will be processed once approved" is a promise about the ladder, not a status on it."""

    @pytest.mark.parametrize(
        "status, claim, expected",
        [
            ("pending_review", "The $66.56 refund will be processed once approved", True),
            ("pending_review", "Your refund for order 10665 will be issued once approved", True),
            ("denied", "The $66.56 refund will be processed once approved", False),
            ("pending_review", "Your refund has been approved and will be processed", False),
            ("issued", "Your refund has been approved and will be processed", True),
        ],
    )
    def test_the_promise(self, status: str, claim: str, expected: bool) -> None:
        assert _holds(status, claim) is expected

    def test_the_rationale_names_the_promise(self) -> None:
        finding = StateModel().judge(
            "The $66.56 refund will be processed once approved", _status("pending_review")
        )
        assert finding is not None and "a promise of" in finding.rationale


class TestACommitmentNeedsAProcessAlreadyUnderWay:
    """ "Will be sent for review" is true only if something has already been sent.

    Order 10533 ($103.39) was answered "this refund request will be sent for
    human review … You'll hear back soon" while its refund status was `none`:
    the model had declined to call the tool whose escalation queues a review,
    so nothing would ever be sent. Reachability alone passed it, because a
    refund at `none` can in principle reach `pending_review`.
    """

    @pytest.mark.parametrize(
        "status, claim, expected",
        [
            ("none", "This refund request will be sent for human review", False),
            ("pending_review", "This refund request will be sent for human review", True),
            ("none", "You'll hear back soon", False),
            ("pending_review", "You'll hear back soon", True),
            ("none", "Your refund for order 10665 will be issued", False),
        ],
    )
    def test_a_commitment(self, status: str, claim: str, expected: bool) -> None:
        assert _holds(status, claim) is expected

    def test_a_conditional_offer_is_not_a_commitment(self) -> None:
        """ "Once you confirm" offers; it does not say anything is under way."""
        assert _holds("none", "Once you confirm, your refund will be issued") is True

    def test_the_rationale_says_nothing_is_under_way(self) -> None:
        finding = StateModel().judge("You'll hear back soon", _status("none"))
        assert finding is not None and "none" in finding.rationale
