"""The HALLUCINATION CHECK card's shared format: every agent's check reads the same.

The support agent's result is the reference shape; the card model must accept it
unchanged, so an agent drawing its own (ops) and the support agent cannot drift apart.
"""

from agent_platform.guardrails.response import ResponseGuardrailResult
from agent_platform.hallucination_check.card import (
    ClaimCheck,
    GroundingCard,
    claim_tally,
    grounding_status,
    outcome_line,
)
from agent_platform.hallucination_check.levels import CheckLevel


def _claims(*verdicts: str, detector: str = "") -> list[dict[str, object]]:
    return [{"claim": f"c{i}", "verdict": v, "detector": detector} for i, v in enumerate(verdicts)]


def test_the_support_agents_result_is_a_grounding_card() -> None:
    result = ResponseGuardrailResult(
        response="Sent.",
        claim_checks=_claims("supported", "undetermined"),
        levels=[CheckLevel(level=1, name="token confidence", verdict="PASS", summary="ok")],
        confidence={"available": True, "sentences": []},
        rule="NLI decides each claim",
    )
    card = GroundingCard.model_validate(result.model_dump(exclude={"response"}))
    assert card.tally == result.tally
    assert card.outcome == result.outcome
    assert [claim.verdict for claim in card.claim_checks] == ["supported", "undetermined"]
    # The agent's extras ride along unchanged.
    assert card.fields()["hallucination_scope"] == "none"


def test_the_tally_counts_each_verdict_in_card_words() -> None:
    tally = claim_tally(_claims("supported", "supported", "undetermined"))
    assert tally == "2 supported · 1 unverified of 3 claims"


def test_one_claim_is_singular_and_none_says_so() -> None:
    assert claim_tally(_claims("refuted")) == "1 contradicted of 1 claim"
    assert claim_tally([]) == "no checkable claims"


def test_a_lexical_fallback_is_flagged_in_the_tally() -> None:
    tally = claim_tally(_claims("supported", detector="lexical-overlap"))
    assert tally.endswith("lexical fallback (NLI model unavailable)")


def test_typed_claims_tally_like_dicts() -> None:
    typed = [
        ClaimCheck(claim="12", verdict="supported"),
        ClaimCheck(claim="4", verdict="undetermined"),
    ]
    assert claim_tally(typed) == claim_tally(_claims("supported", "undetermined"))


def test_the_result_line_says_what_was_sent() -> None:
    assert outcome_line(
        escalated=False, withheld=False, removed=0, total=2, reason_code=""
    ).startswith("PASSED")
    trimmed = outcome_line(
        escalated=True, withheld=False, removed=1, total=3, reason_code="LOW_CONFIDENCE_SPECIFIC"
    )
    assert trimmed.startswith("TRIMMED · removed 1 of 3")
    assert "low confidence" in trimmed
    withheld = outcome_line(escalated=True, withheld=True, removed=2, total=2, reason_code="X")
    assert withheld.startswith("BLOCKED")


def test_the_card_is_coloured_by_what_was_sent() -> None:
    assert grounding_status(withheld=False, escalated=False) == "completed"
    assert grounding_status(withheld=False, escalated=True) == "warning"
    assert grounding_status(withheld=True, escalated=True) == "blocked"
