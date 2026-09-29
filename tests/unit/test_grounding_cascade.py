"""The tiered grounding cascade and its claim classes (LLD §10).

Every case here is a reply a customer should have received and did not:
a correct refund confirmation, a scripted "you'll hear back soon", a
polite closing. They are written against the evidence the agent really
builds — the tool's own sentences, the knowledge article, the policy rule
— so a regression reads as a real reply being blocked.
"""

import pytest

from agentic_commons.context.types import Chunk
from agentic_commons.evals import create_scoring_cascade
from agentic_commons.guardrails.response import ResponseGuardrail
from agentic_commons.hallucination_protection import (
    ClaimClassifier,
    ClaimDecomposer,
    DetectionLadder,
)

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
