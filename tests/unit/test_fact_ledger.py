"""The fact ledger: claims read against typed facts before any model reads them.

Each case is a sentence a live run got wrong while the evidence that decides it was
in hand (a delivery date, a settlement window, the verified customer, an eligibility
flag, a limit to compare against).
"""

import pytest

from agent_platform.config.settings import FactAttributeSettings, FactSettings
from agent_platform.context.types import Chunk
from agent_platform.hallucination_protection.ledger import (
    FLAG_TRUE,
    Fact,
    FactLedger,
    fact_chunk,
    facts_in,
)

_SETTINGS = FactSettings(
    attributes={
        "order_status": FactAttributeSettings(type="text"),
        "delivered_on": FactAttributeSettings(type="date", terms=[r"\bdeliver"]),
        "order_total": FactAttributeSettings(type="money", terms=[r"\btotal", r"\brefund"]),
        "refund_limit": FactAttributeSettings(type="money", terms=[r"\blimit\b"]),
        "settlement_window": FactAttributeSettings(type="duration", terms=[r"\bsettl"]),
        "refund_eligible": FactAttributeSettings(
            type="flag",
            affirm=[r"\beligible\s+for\s+(?:a\s+)?refund"],
            deny=[r"\bnot\s+eligible\b", r"\bcannot\s+be\s+refunded\b"],
        ),
        "customer_name": FactAttributeSettings(type="text"),
        "customer_id": FactAttributeSettings(type="number", terms=[r"\baccount\b"]),
        "verified": FactAttributeSettings(type="flag", affirm=[r"\bverified\b"]),
    },
    comparisons={">": [r"\bexceeds?\b"], "<=": [r"\bwithin\b"]},
    duration_units={"minutes?": 60, "hours?": 3600},
)


def _fact(attribute: str, value: str, subject: str = "10986") -> Chunk:
    return fact_chunk(
        Fact(
            subject=subject,
            attribute=attribute,
            value=value,
            statement=f"{attribute} of {subject or 'the session'} is {value}.",
            kind="tool_result" if subject else "policy",
            source=f"test:{attribute}",
        )
    )


_ORDER = [
    _fact("order_status", "delivered"),
    _fact("delivered_on", "2026-08-02"),
    _fact("order_total", "$67.27"),
    _fact("refund_eligible", FLAG_TRUE),
]
_RULES = [_fact("refund_limit", "$50.00", ""), _fact("settlement_window", "3600", "")]
_SESSION = [
    _fact("customer_name", "Dennis Johnson", ""),
    _fact("customer_id", "2", ""),
    _fact("verified", FLAG_TRUE, ""),
]


def _judge(claim: str, evidence: list[Chunk]) -> bool | None:
    finding = FactLedger(_SETTINGS, identifier=r"\b\d{5}\b").judge(claim, evidence)
    return None if finding is None else finding.holds


class TestTheFactsDecide:
    @pytest.mark.parametrize(
        "claim, evidence",
        [
            ("Your order 10986 was delivered on 2026-08-02", _ORDER),
            ("Your order 10986 is eligible for a refund", _ORDER),
            ("It will settle within 60 minutes", _RULES),
            ("You're Dennis Johnson, verified customer with account #2", _SESSION),
            ("The refund amount of $67.27 exceeds the $50.00 limit", [*_ORDER, *_RULES]),
        ],
    )
    def test_what_the_record_states_is_supported(self, claim: str, evidence: list[Chunk]) -> None:
        assert _judge(claim, evidence) is True

    @pytest.mark.parametrize(
        "claim, evidence",
        [
            ("Your order 10986 was delivered on 2026-08-05", _ORDER),
            ("Your order 10986 is not eligible for a refund", _ORDER),
            ("It will settle within 2 hours", _RULES),
            ("The refund amount of $50.00 exceeds the $67.27 total", [*_ORDER, *_RULES]),
        ],
    )
    def test_what_the_record_disagrees_with_is_refuted(
        self, claim: str, evidence: list[Chunk]
    ) -> None:
        assert _judge(claim, evidence) is False


class TestTheLedgerKeepsQuietWhenItCannotTell:
    def test_a_detail_no_fact_accounts_for_goes_on_to_the_next_check(self) -> None:
        """Delivered on the record's date, but "tracking 4471" is in no fact."""
        claim = "Your order 10986 was delivered on 2026-08-02 with tracking 4471"
        assert _judge(claim, _ORDER) is None

    def test_a_claim_with_no_facts_in_evidence_is_not_judged(self) -> None:
        assert _judge("Your order 10986 was delivered", []) is None

    def test_a_promise_is_not_supported_by_the_status_it_names(self) -> None:
        assert _judge("Your order 10986 will be delivered", _ORDER) is None

    def test_a_denied_status_is_not_supported_by_the_status(self) -> None:
        assert _judge("Your order 10986 has not been delivered", _ORDER) is None


class TestWhichFactsApply:
    def test_another_orders_facts_do_not_speak_for_this_one(self) -> None:
        other = [_fact("delivered_on", "2026-08-02", "10854")]
        assert _judge("Your order 10986 was delivered on 2026-08-05", other) is None

    def test_a_claim_naming_no_order_reads_the_one_order_in_play(self) -> None:
        assert _judge("It was delivered on 2026-08-02", _ORDER) is True

    def test_with_two_orders_in_play_an_unnamed_claim_reads_only_global_facts(self) -> None:
        two = [*_ORDER, _fact("delivered_on", "2026-08-09", "10854")]
        assert _judge("It was delivered on 2026-08-02", two) is None


class TestFactsTravelAsEvidence:
    def test_a_fact_chunk_reads_as_its_sentence_and_its_value(self) -> None:
        [(fact, chunk)] = facts_in([_fact("order_total", "$67.27")])
        assert chunk.content == "order_total of 10986 is $67.27."
        assert (fact.attribute, fact.value, fact.subject) == ("order_total", "$67.27", "10986")

    def test_a_plain_chunk_carries_no_fact(self) -> None:
        assert facts_in([Chunk(content="Order 10986 is delivered", kind="tool_result")]) == []
