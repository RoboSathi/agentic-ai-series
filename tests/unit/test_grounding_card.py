"""The HALLUCINATION CHECK card: the result, the model's confidence, one line per claim.

What it no longer draws matters as much as what it does: the claim tally is the
title, and the evidence counts, the draft check and the id/amount row are in the
log, not repeated on the card.
"""

from typing import Any

from rich.text import Text

from agent_platform.config import get_settings
from agent_platform.tui.widgets import _append_grounding

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
        assert "confidence · test-model" in card
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
