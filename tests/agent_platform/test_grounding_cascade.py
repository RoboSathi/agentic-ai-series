"""The tiered grounding cascade and its claim classes (LLD §10).

Every case here is a reply a customer should have received and did not:
a correct refund confirmation, a scripted "you'll hear back soon", a
polite closing. They are written against the evidence the agent really
builds — the tool's own sentences, the knowledge article, the policy rule
— so a regression reads as a real reply being blocked.
"""

import pytest

from agent_platform.context.types import Chunk
from agent_platform.guardrails.response import ResponseGuardrail, ResponseGuardrailResult
from agent_platform.hallucination_check import (
    ClaimClassifier,
    ClaimDecomposer,
    DetectionLadder,
)
from agent_platform.scoring import create_scoring_cascade
from agent_platform.scoring.entailment import EntailmentResult

_ORDER = 10864


def _tool(*facts: str) -> list[Chunk]:
    return [
        Chunk(content=fact, kind="tool_result", metadata={"observation_sequence": index})
        for index, fact in enumerate(facts)
    ]


_COMPLETED = _tool(
    f"The refund of $18.70 for order {_ORDER} has been completed",
    f"Order {_ORDER} refund status: completed",
)
_REFERRED = _tool(f"Refund of $18.70 for order {_ORDER} has been queued for human review")
_ARTICLE = Chunk(
    content="Once a refund is issued, it goes back to the original payment method.",
    kind="policy",
    source="knowledge:refunds.md#3",
)
_SCRIPT = Chunk(
    content=(
        "Some requests are sent to a person before any money moves. Tell the customer "
        "their request needs a manual review and they'll hear back soon."
    ),
    kind="policy",
    source="refund_policy#3",
)
_POLITE = Chunk(
    content="Be warm and courteous, and thank the customer for getting in touch.",
    kind="system",
    source="system_prompt#1",
)


def _guard() -> ResponseGuardrail:
    return ResponseGuardrail(DetectionLadder(create_scoring_cascade()))


class TestClaimClasses:
    @pytest.mark.parametrize(
        "claim",
        [
            "Thanks for reaching out",
            "I'm happy to help with that",
            "Is there anything else I can do for you",
            "Sorry for the wait",
        ],
    )
    def test_conversation_is_not_a_claim(self, claim: str) -> None:
        assert ClaimClassifier().classify(claim) == "conversational"

    @pytest.mark.parametrize(
        "claim",
        [
            "Order 10864 has been refunded",
            "$18.70 has been refunded",
            "The refund has been completed",
        ],
    )
    def test_a_fact_about_the_case_is_a_record_claim(self, claim: str) -> None:
        assert ClaimClassifier().classify(claim) == "record"

    @pytest.mark.parametrize(
        "claim",
        [
            "You will hear back soon",
            "Your request will be processed shortly",
            "Refunds go back to the original payment method",
            "I'll get this queued for review",
        ],
    )
    def test_a_promise_or_a_rule_is_a_rule_claim(self, claim: str) -> None:
        """Future and modal wording promises; it does not report an outcome."""
        assert ClaimClassifier().classify(claim) == "rule"

    def test_a_thank_you_that_carries_a_fact_is_still_a_record_claim(self) -> None:
        assert ClaimClassifier().classify("Thanks, order 10864 has been cancelled") == "record"


class TestAMixedClaimIsSplitByClass:
    def test_the_destination_is_carved_out_of_the_outcome(self) -> None:
        claims = ClaimDecomposer().decompose(
            "$18.70 has been refunded to your original payment method."
        )
        assert [(c.text, c.claim_class) for c in claims] == [
            ("$18.70 has been refunded", "record"),
            ("A refund goes back to the original payment method", "rule"),
        ]


@pytest.mark.usefixtures("abstaining_guard")
class TestTheReportedReplies:
    async def test_a_completed_refund_and_where_it_went(self) -> None:
        reply = "$18.70 has been refunded to your original payment method."
        result = await _guard().enforce(reply, [*_COMPLETED, _ARTICLE, _SCRIPT, _POLITE])
        assert not result.escalated
        assert result.response == reply
        assert result.unsupported == 0

    async def test_you_will_hear_back_soon_on_a_referral(self) -> None:
        reply = "Your request needs a manual review and you'll hear back soon."
        result = await _guard().enforce(reply, [*_REFERRED, _ARTICLE, _SCRIPT, _POLITE])
        assert not result.escalated
        assert result.response == reply

    async def test_the_whole_reply_from_the_report(self) -> None:
        reply = (
            "1. $18.70 has been refunded to your original payment method. "
            "2. You will hear back soon."
        )
        result = await _guard().enforce(reply, [*_REFERRED, _ARTICLE, _SCRIPT, _POLITE])
        # A referral does not settle a refund: claim 1 is refuted, and only it.
        assert result.escalated
        assert "hear back soon" in result.response
        assert "refunded" not in result.response

    async def test_a_polite_closing_needs_no_evidence(self) -> None:
        reply = "Thanks for getting in touch. Happy to help with anything else."
        result = await _guard().enforce(reply, [*_COMPLETED, _POLITE])
        assert not result.escalated

    async def test_a_polite_reply_on_a_turn_with_no_tool_is_not_a_business_claim(self) -> None:
        """ "…refund processed" as a hope is a promise, not an outcome."""
        reply = "Happy to help - I'll get your refund processed once you confirm the order."
        result = await _guard().enforce(reply, [_POLITE, _ARTICLE])
        assert not result.escalated


@pytest.mark.usefixtures("abstaining_guard")
class TestTheGuardStillGuards:
    async def test_a_wrong_amount_is_refuted(self) -> None:
        result = await _guard().enforce(
            f"$99.00 has been refunded for order {_ORDER}.", [*_COMPLETED, _ARTICLE]
        )
        assert result.escalated

    async def test_an_outcome_with_no_tool_result_is_blocked(self) -> None:
        result = await _guard().enforce("I've processed your refund of $18.70.", [_ARTICLE])
        assert result.escalated
        assert result.reason_code == "BUSINESS_CLAIM_WITHOUT_TOOL_EVIDENCE"

    async def test_a_referral_is_not_a_settled_refund(self) -> None:
        result = await _guard().enforce(
            f"The refund for order {_ORDER} has been completed.", _REFERRED
        )
        assert result.escalated

    async def test_a_policy_line_cannot_prove_an_outcome(self) -> None:
        """Only a tool result or a verification can show something happened."""
        result = await _guard().enforce(
            "Your refund has been issued.", [_ARTICLE, _SCRIPT, _POLITE]
        )
        assert result.escalated

    async def test_a_rule_nobody_stated_is_unverified_not_refuted(self) -> None:
        """No evidence either way is an abstention, and abstaining never blocks."""
        result = await _guard().enforce("Refunds take a while to arrive.", [_ARTICLE])
        assert not result.escalated
        assert result.undetermined == 1


@pytest.mark.usefixtures("abstaining_guard")
class TestFailClosedKeepsWhatWasNotRefuted:
    async def test_an_unverified_sentence_survives_a_refuted_neighbour(self) -> None:
        reply = (
            f"$99.00 has been refunded for order {_ORDER}. "
            "Refunds take a while to arrive. Thanks for your patience."
        )
        result = await _guard().enforce(reply, [*_COMPLETED, _ARTICLE])
        assert result.escalated
        assert "$99.00" not in result.response
        assert "Refunds take a while to arrive." in result.response
        assert "Thanks for your patience." in result.response
        assert result.unsupported == 1


class TestASentenceGoesWhenAnyClaimInItIsRefuted:
    async def test_a_refuted_carved_rule_takes_its_whole_sentence_with_it(self) -> None:
        """The carved rule is restated in the config's words, which are not in the reply."""
        contrary = Chunk(
            content="Refunds are never returned to the original payment method.",
            kind="policy",
            source="refund_policy#9",
        )
        reply = (
            "$18.70 has been refunded to your original payment method. Thanks for your patience."
        )
        result = await _guard().enforce(reply, [*_COMPLETED, contrary])
        assert result.escalated
        assert result.response == "Thanks for your patience."


_REFUSED = _tool(
    f"NOT EXECUTED support__process_refund(reason=Customer requested refund, order_id={_ORDER}, "
    f"amount=90.93): 'process_refund' names order:{_ORDER}, which is not a record this "
    "conversation established"
)


class TestARefusalIsGroundedByTheCallThatWasRefused:
    """NLI cannot relate "I'm unable to…" to a structured NOT EXECUTED line; the record can."""

    async def test_an_inability_the_record_shows_is_supported(self) -> None:
        reply = f"I'm unable to process a refund for order {_ORDER} at this time."
        result = await _guard().enforce(reply, _REFUSED)
        assert not result.escalated
        assert result.supported == 1
        assert result.claim_checks[0]["decided_by"] == "deterministic_provenance"

    async def test_the_same_words_are_not_supported_when_the_refund_went_through(self) -> None:
        reply = f"I'm unable to process a refund for order {_ORDER} at this time."
        result = await _guard().enforce(reply, _COMPLETED)
        assert result.supported == 0

    async def test_a_refusal_about_another_order_is_not_supported_by_this_one(self) -> None:
        result = await _guard().enforce("I'm unable to process a refund for order 99999.", _REFUSED)
        assert result.escalated


@pytest.mark.usefixtures("abstaining_guard")
class TestAnUnverifiedClaimSaysWhy:
    async def test_each_tier_gives_its_reason(self) -> None:
        result = await _guard().enforce("Refunds take a while to arrive.", [_ARTICLE, _POLITE])
        why = result.claim_checks[0]["rationale"]
        assert result.undetermined == 1
        assert "policy:" in why and "system:" in why and "memory: no evidence" in why


class TestTierOrder:
    async def test_the_system_prompt_never_refutes(self) -> None:
        instruction = Chunk(
            content="Do not mention refunds unless the customer raised the subject.",
            kind="system",
            source="system_prompt#4",
        )
        result = await _guard().enforce(
            f"Order {_ORDER} has been refunded.", [*_COMPLETED, instruction]
        )
        assert not result.escalated

    async def test_memory_never_refutes(self) -> None:
        memory = Chunk(
            content=f"Order {_ORDER} has not been refunded.",
            kind="memory",
            source="memory:semantic:1",
        )
        result = await _guard().enforce(f"Order {_ORDER} has been refunded.", [*_COMPLETED, memory])
        assert not result.escalated

    async def test_the_customers_own_words_are_not_evidence(self) -> None:
        said = Chunk(
            content=f"Order {_ORDER} has been refunded.", kind="history", source="customer_message"
        )
        result = await _guard().enforce(f"Order {_ORDER} has been refunded.", [*_REFERRED, said])
        assert result.escalated


class _Neutral:
    """The hosted model's answer to "returned" against "Refunded": neutral, and sure of it."""

    model_name = "stub-nli"

    async def entails(self, premise: str, hypothesis: str) -> EntailmentResult:
        return EntailmentResult(
            label="unknown", entailed=False, score=0.99, calibrated=True, model="stub-nli"
        )


def _hosted_guard() -> ResponseGuardrail:
    return ResponseGuardrail(DetectionLadder(create_scoring_cascade(), entailment=_Neutral()))


class _Scripted:
    """The hosted model on order 10169: sure the stale read contradicts, neutral on the rest."""

    model_name = "stub-nli"

    async def entails(self, premise: str, hypothesis: str) -> EntailmentResult:
        stale = premise.startswith("No refund has been requested")
        return EntailmentResult(
            label="contradiction" if stale else "unknown",
            entailed=False,
            score=1.0 if stale else 0.99,
            calibrated=True,
            model="stub-nli",
        )


def _scripted_guard() -> ResponseGuardrail:
    return ResponseGuardrail(DetectionLadder(create_scoring_cascade(), entailment=_Scripted()))


def _seen(text: str, sequence: int, *, wrote: bool = False, source: str = "") -> Chunk:
    return Chunk(
        content=text,
        kind="tool_result",
        source=source,
        metadata={"observation_sequence": sequence, "state_changing": wrote},
    )


class TestAReadBeforeAWriteIsStale:
    """Order 10169: the status was read (none), then the refund was referred (pending review)."""

    _REPLY = (
        "Your refund request for order 10169 has been received and is currently pending review."
    )
    _READ = _seen("No refund has been requested for order 10169.", 0)
    _WRITE = [
        _seen("Refund of $73.63 for order 10169 has been queued for human review", 1, wrote=True),
        _seen("Order 10169 refund status: pending_review.", 1, wrote=True),
    ]

    async def test_the_world_after_the_write_is_the_world(self) -> None:
        result = await _scripted_guard().enforce(self._REPLY, [self._READ, *self._WRITE])
        assert not result.escalated, result.claim_checks
        assert result.response == self._REPLY

    async def test_a_turn_that_only_read_keeps_every_read(self) -> None:
        result = await _scripted_guard().enforce(self._REPLY, [self._READ])
        assert result.escalated

    async def test_a_write_on_another_order_does_not_make_this_read_stale(self) -> None:
        read = _seen("No refund has been requested for order 10001.", 0)
        result = await _scripted_guard().enforce(
            "Your refund request for order 10001 has been received.", [read, *self._WRITE]
        )
        assert result.escalated


@pytest.mark.usefixtures("abstaining_guard")
class TestASettledRefundIsSupportedByTheRecord:
    """The amount was in the record and the sentence was still "unverified"."""

    _SETTLED = [
        _seen("Refunded $21.35 for order 10001.", 0, wrote=True),
        _seen("Order 10001 refund status: completed", 0, wrote=True),
    ]
    _FULL = Chunk(
        content="A refund is always for the **full order total**.",
        kind="policy",
        source="refund_policy#1",
    )
    _REPLY = (
        "Your refund for order 10001 has been processed. The full amount of $21.35 was "
        "returned and the refund status is now completed."
    )

    async def test_the_live_reply_has_nothing_unverified(self) -> None:
        result = await _hosted_guard().enforce(self._REPLY, [*self._SETTLED, self._FULL])
        assert not result.escalated, result.claim_checks
        assert result.undetermined == 0
        assert result.supported == result.claims

    async def test_the_card_says_what_the_record_showed(self) -> None:
        result = await _hosted_guard().enforce(
            "The full amount of $21.35 was returned.", [*self._SETTLED, self._FULL]
        )
        why = next(c["rationale"] for c in result.claim_checks if "21.35" in c["claim"])
        assert "completed" in why and "21.35" in why

    async def test_a_wrong_amount_is_still_contradicted(self) -> None:
        result = await _hosted_guard().enforce(
            "The full amount of $99.00 was returned.", [*self._SETTLED, self._FULL]
        )
        assert result.escalated

    async def test_a_referral_is_not_a_returned_amount(self) -> None:
        queued = [_seen("Refund of $21.35 for order 10001 has been queued for human review", 0)]
        result = await _hosted_guard().enforce(
            "The full amount of $21.35 was returned.", [*queued, self._FULL]
        )
        assert result.escalated

    async def test_a_returned_item_is_not_a_refund_claim(self) -> None:
        result = await _hosted_guard().enforce("Your item was returned to the warehouse.", [])
        assert not result.escalated

    async def test_a_neutral_verdict_is_named_neutral(self) -> None:
        result = await _hosted_guard().enforce(
            "Once a refund is issued it takes seven weeks to arrive.", [_ARTICLE]
        )
        why = result.claim_checks[0]["rationale"]
        assert "neutral" in why and "unknown" not in why


class _SiblingsDisagree:
    """One observation, two sentences, and a model that contradicts the closer one."""

    model_name = "stub-nli"

    async def entails(self, premise: str, hypothesis: str) -> EntailmentResult:
        contradicts = premise.endswith(": pending")
        return EntailmentResult(
            label="contradiction" if contradicts else "entailment",
            entailed=not contradicts,
            score=0.99,
            calibrated=True,
            model="stub-nli",
        )


class TestAnObservationIsOneConsistentRecord:
    _CLAIM = "Order 10665 was dispatched by courier."
    _ENTAILS = "Courier dispatched order 10665 yesterday."
    _CONTRADICTS = "Order 10665 was dispatched by courier: pending"

    def _guard(self) -> ResponseGuardrail:
        ladder = DetectionLadder(create_scoring_cascade(), entailment=_SiblingsDisagree())
        return ResponseGuardrail(ladder)

    async def test_a_sentence_that_entails_beats_a_sibling_that_contradicts(self) -> None:
        evidence = [_seen(self._ENTAILS, 0), _seen(self._CONTRADICTS, 0)]
        result = await self._guard().enforce(self._CLAIM, evidence)
        assert not result.escalated, result.claim_checks

    async def test_a_contradiction_with_no_entailing_sibling_still_refutes(self) -> None:
        result = await self._guard().enforce(self._CLAIM, [_seen(self._CONTRADICTS, 0)])
        assert result.escalated


class _ByPremise:
    """A model with a fixed opinion on each premise, neutral on any it was not given."""

    model_name = "stub-nli"

    def __init__(self, labels: dict[str, str]) -> None:
        self._labels = labels

    async def entails(self, premise: str, hypothesis: str) -> EntailmentResult:
        label = self._labels.get(premise, "unknown")
        return EntailmentResult(
            label=label,
            entailed=label == "entailment",
            score=0.99,
            calibrated=True,
            model="stub-nli",
        )


def _judged_by(labels: dict[str, str]) -> ResponseGuardrail:
    return ResponseGuardrail(
        DetectionLadder(create_scoring_cascade(), entailment=_ByPremise(labels))
    )


def _all_supported(result: ResponseGuardrailResult) -> bool:
    return not result.escalated and result.claims > 0 and result.supported == result.claims


class TestTheCurrentEvidenceIsOneConsistentWorld:
    """Order 10842: cancelled, its refund approved — and "was cancelled" was blocked.

    The order read entailed the claim; the refund read, taken later and about
    another attribute, "contradicted" it at 0.99, and the later read won.
    Reads that nothing has overtaken describe one world, so they cannot
    disagree about it: a contradiction stands only when no current evidence
    entails the claim.
    """

    _CLAIM = "Order 10842 (Bluetooth Speaker) was cancelled."
    _ORDER = "Order 10842 (Bluetooth Speaker) has been cancelled, so it is not on its way"
    _REFUND = "The refund for order 10842 has been approved."

    async def test_a_later_read_of_another_attribute_cannot_refute(self) -> None:
        guard = _judged_by({self._ORDER: "entailment", self._REFUND: "contradiction"})
        evidence = [
            _seen(self._ORDER, 0, source="get_order_status"),
            _seen(self._REFUND, 1, source="get_refund_status"),
        ]
        assert _all_supported(await guard.enforce(self._CLAIM, evidence))

    async def test_the_order_of_the_reads_does_not_matter(self) -> None:
        guard = _judged_by({self._ORDER: "entailment", self._REFUND: "contradiction"})
        evidence = [
            _seen(self._REFUND, 0, source="get_refund_status"),
            _seen(self._ORDER, 1, source="get_order_status"),
        ]
        assert _all_supported(await guard.enforce(self._CLAIM, evidence))

    async def test_a_claim_nothing_entails_is_still_refuted(self) -> None:
        guard = _judged_by({self._ORDER: "contradiction", self._REFUND: "contradiction"})
        evidence = [
            _seen(self._ORDER, 0, source="get_order_status"),
            _seen(self._REFUND, 1, source="get_refund_status"),
        ]
        assert (await guard.enforce(self._CLAIM, evidence)).escalated

    async def test_a_later_read_from_the_same_tool_overtakes_the_earlier(self) -> None:
        """Re-read after the state moved: the stale read may no longer support."""
        before = "Order 10842 (Bluetooth Speaker) is waiting to be cancelled"
        after = "Order 10842 (Bluetooth Speaker) has been delivered"
        guard = _judged_by({before: "entailment", after: "contradiction"})
        evidence = [
            _seen(before, 0, source="get_order_status"),
            _seen(after, 1, source="get_order_status"),
        ]
        assert (await guard.enforce(self._CLAIM, evidence)).escalated

    async def test_reads_of_unknown_origin_never_overtake_each_other(self) -> None:
        """An unnamed source is no tool, so two such reads are not a re-read."""
        after = "Order 10842 (Bluetooth Speaker) has been delivered"
        guard = _judged_by({self._ORDER: "entailment", after: "contradiction"})
        evidence = [_seen(self._ORDER, 0), _seen(after, 1)]
        assert _all_supported(await guard.enforce(self._CLAIM, evidence))

    async def test_a_refused_re_read_overtakes_nothing(self) -> None:
        """A call that was refused observed nothing, so the earlier read still stands."""
        refused = "NOT EXECUTED get_order_status(order_id=10842): budget exhausted"
        guard = _judged_by({self._ORDER: "entailment"})
        evidence = [
            _seen(self._ORDER, 0, source="get_order_status"),
            _seen(refused, 1, source="get_order_status"),
        ]
        assert _all_supported(await guard.enforce(self._CLAIM, evidence))

    async def test_a_later_read_of_another_order_overtakes_nothing(self) -> None:
        other = "Order 10843 (Desk Lamp) has been delivered"
        guard = _judged_by({self._ORDER: "entailment", other: "contradiction"})
        evidence = [
            _seen(self._ORDER, 0, source="get_order_status"),
            _seen(other, 1, source="get_order_status"),
        ]
        assert _all_supported(await guard.enforce(self._CLAIM, evidence))

    async def test_another_orders_record_cannot_support(self) -> None:
        """Entailment must be about what the claim names, as a refutation must."""
        other = "Order 10843 (Bluetooth Speaker) has been cancelled"
        guard = _judged_by({other: "entailment"})
        result = await guard.enforce(self._CLAIM, [_seen(other, 0, source="get_order_status")])
        assert result.supported == 0, result.claim_checks
