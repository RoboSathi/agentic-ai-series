"""The HALLUCINATION CHECK card: the result, the model's confidence, one line per claim.

What it no longer draws matters as much as what it does: the claim tally is the
title, and the evidence counts, the draft check and the id/amount row are in the
log, not repeated on the card.
"""

from typing import Any

from rich.text import Text

from agent_platform.config import get_settings
from agent_platform.interface.tui.widgets import _append_grounding

_SENTENCE = "Order 10986 totals $67.27."


def _fields(**extra: Any) -> dict[str, Any]:
    return {
        "outcome": "TRIMMED · removed 1 of 2 sentence(s) · states something no evidence supports",
        "tally": "1 supported · 1 unsupported of 2 claims",
        "pre_check": "passed → send_draft · evidence: order_record:10986",
        "evidence_searched": "4 tool result · 32 policy",
        "provenance_checks": ["✓ order 10986 — tool result"],
        "confidence": {
            "available": True,
            "model": "test-model",
            "sentences": [
                {
                    "sentence": _SENTENCE,
                    "content_mean": 0.42,
                    "tokens": [
                        {"start": 0, "end": 5, "prob": 0.99, "slot": False},
                        {"start": 19, "end": 21, "prob": 0.04, "slot": True},
                    ],
                }
            ],
        },
        "claim_checks": [
            {
                "claim": "Order 10986 totals $67.27",
                "verdict": "supported",
                "method": "record",
                "rationale": "order_total is $67.27",
                "evidence": [
                    {
                        "tier": "tool result",
                        "source": "order_record:10986",
                        "excerpt": "Order 10986 totals $67.27.",
                        "nli_label": "record",
                        "decisive": True,
                    }
                ],
            },
            {
                "claim": "A reviewer will call you tonight",
                "verdict": "unsupported",
                "method": "no evidence",
                "evidence": [],
            },
        ],
        **extra,
    }


def _card(fields: dict[str, Any]) -> Text:
    body = Text()
    _append_grounding(body, fields)
    return body


class TestWhatTheCardSays:
    def test_the_result_the_confidence_and_each_claim(self) -> None:
        card = _card(_fields()).plain
        assert "result = TRIMMED · removed 1 of 2" in card
        # Logprobs first, in one line: the mean, and the least sure sentence.
        assert "logprobs · test-model = 42% mean · lowest 42% (sentence 1)" in card
        assert f"42% {_SENTENCE}" in card
        assert "✓ Order 10986 totals $67.27 [SUPPORTED] · record" in card
        assert (
            'supported by [Tool Result · order_record:10986]: "Order 10986 totals $67.27."' in card
        )
        assert "✗ A reviewer will call you tonight [UNSUPPORTED] · no evidence" in card
        assert "unsupported: no evidence states this" in card

    def test_what_the_title_and_the_log_already_say_is_not_repeated(self) -> None:
        card = _card(_fields()).plain
        for gone in ("pre-check", "checked against", "ids & amounts", "checker", "1 supported ·"):
            assert gone not in card

    def test_a_correction_says_what_was_put_back(self) -> None:
        card = _card(_fields(corrected_sentences=["Order 10226's refund is issued."])).plain
        assert "corrected from the record = Order 10226's refund is issued." in card


class TestTheConfidenceIsColoured:
    def _style_of(self, card: Text, text: str) -> str:
        at = card.plain.index(text)
        return " ".join(str(span.style) for span in card.spans if span.start <= at < span.end)

    def test_a_guessed_sentence_reads_as_a_failure(self) -> None:
        card = _card(_fields())
        assert get_settings().tui.tone_styles["fail"] in self._style_of(card, "42%")

    def test_a_guessed_specific_is_marked_in_its_sentence(self) -> None:
        card = _card(_fields())
        style = self._style_of(card, "67")
        assert get_settings().tui.tone_styles["fail"] in style and "underline" in style

    def test_an_agent_without_token_confidences_shows_its_levels(self) -> None:
        level = {"level": 1, "name": "token confidence", "verdict": "SKIPPED", "summary": "none"}
        card = _card(_fields(confidence={}, levels=[level])).plain
        assert "level 1 · token confidence = none" in card


def _steps_card() -> Text:
    """A claim NLI could not decide and the second opinion confirmed, and one NLI refuted."""
    page = "robosathi.com/about/"
    claims = [
        {
            "claim": "Kumar Karan was an Engineering Manager at Turing",
            "verdict": "supported",
            "readings": [
                {"step": 1, "check": "NLI DeBERTa", "result": "neutral 1.000",
                 "decision": "not decided", "tone": "warn",
                 "evidence": [{"where": page, "excerpt": "Kumar Karan"}]},
                {"step": 2, "check": "Laya", "result": "supported 0.512 ≥ 0.45",
                 "decision": "confirmed", "tone": "pass",
                 "evidence": [{"where": page,
                               "excerpt": "Kumar Karan\nEngineering Manager — AI @ Turing\nBlog"}]},
            ],
        },
        {
            "claim": "Kumar Karan was a Lead Instructor at Turing",
            "verdict": "refuted",
            "readings": [
                {"step": 1, "check": "NLI DeBERTa", "result": "contradiction 0.999",
                 "decision": "contradicted", "tone": "fail",
                 "evidence": [{"where": page, "excerpt": "Lead Instructor @ Scaler"}]},
            ],
        },
        {
            "claim": "Your refund for order 10677 is now pending review",
            "verdict": "supported",
            "readings": [
                {"step": 1, "check": "record", "decision": "supported", "tone": "pass",
                 "evidence": [
                     {"where": "DB record · order 10677", "excerpt": "Order 10677 totals $95.86."},
                     {"where": "Tool result · get_refund_status",
                      "excerpt": "Order 10677 refund status: pending_review"},
                 ]},
            ],
        },
    ]  # fmt: skip
    return _card(_fields(claim_checks=claims, rule="NLI DeBERTa first · Laya only where …"))


class TestEachClaimSaysWhichModelDecidedIt:
    def test_nli_then_the_second_opinion_in_order(self) -> None:
        card = _steps_card().plain
        assert "check = NLI DeBERTa first · Laya only where …" in card
        # The step that did not decide says so; the one that decided says only what it read.
        assert (
            '├ NLI DeBERTa neutral 1.000 → not decided · robosathi.com/about/: "Kumar Karan"'
            in card
        )
        assert "└ Laya supported 0.512 ≥ 0.45 · robosathi.com/about/" in card
        assert "→ confirmed" not in card
        # A model that read several lines is quoted by the one closest to the claim.
        assert 'read 3 lines · "Engineering Manager — AI @ Turing"' in card

    def test_a_model_the_claim_never_reached_is_not_named(self) -> None:
        card = _steps_card().plain
        refuted = card[card.index("Lead Instructor at Turing") :].split("✓")[0]
        assert "Laya" not in refuted

    def test_every_line_that_decided_it_is_named_by_kind_and_source(self) -> None:
        card = _steps_card().plain
        assert '├ DB record · order 10677: "Order 10677 totals $95.86."' in card
        assert (
            '└ Tool result · get_refund_status: "Order 10677 refund status: pending_review"' in card
        )
        assert "robosathi.com/about/" in card

    def test_the_colour_is_the_decisions_never_the_claims(self) -> None:
        card = _steps_card()
        tones, styles = get_settings().tui.tone_styles, get_settings().tui.body_styles

        def style_of(fragment: str) -> str:
            start = card.plain.index(fragment)
            return next(str(s.style) for s in card.spans if s.start <= start < s.end)

        assert style_of("Kumar Karan was an Engineering") == styles["value"]
        assert style_of("NLI DeBERTa neutral 1.000") == tones["warn"]
        assert style_of("Laya supported 0.512") == tones["pass"]
        assert style_of("NLI DeBERTa contradiction 0.999") == tones["fail"]
        assert style_of("DB record · order 10677") == styles["value"]
