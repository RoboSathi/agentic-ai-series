"""Level 1 decides on its own: a guessed sentence, or a guessed specific, is cut.

Logprobs say how sure the model was, not whether it was right, so they cut only
what a model writes when it is guessing: a sentence whose content it was unsure of
throughout, and an id, amount or date it was unsure of that no record confirms. A
dip on a grammar word is another phrasing, not a guess, and changes nothing.
"""

import math
from collections.abc import Iterator

import pytest

from agent_platform.evals import create_scoring_cascade
from agent_platform.guardrails import ResponseGuardrail
from agent_platform.hallucination_protection import DetectionLadder
from agent_platform.hallucination_protection.confidence import (
    LOW_CONFIDENCE_DEGENERATE,
    LOW_CONFIDENCE_SPECIFIC,
    ConfidenceAssessment,
    ConfidenceCheck,
    SentenceConfidence,
    gate,
)
from agent_platform.hallucination_protection.types import DetectionVerdict, GenerationAssessment
from agent_platform.observability.unit_context import (
    Generation,
    TaskBinding,
    TurnState,
    bound_turn,
    remember_generation,
)


def _tokens(*pairs: tuple[str, float]) -> list[tuple[str, float]]:
    return [(token, math.log(prob)) for token, prob in pairs]


@pytest.fixture
def turn() -> Iterator[TurnState]:
    state = TurnState(
        turn_id="turn", session_id="s", task=TaskBinding(task_id="t", kind="k", implicit=True)
    )
    with bound_turn(state):
        yield state


def _generated(*pairs: tuple[str, float]) -> None:
    tokens = _tokens(*pairs)
    remember_generation(Generation(model="m", content="".join(t for t, _ in tokens), tokens=tokens))


class TestScoring:
    def test_a_guessed_sentence_is_degenerate(self, turn: TurnState) -> None:
        _generated(("The", 0.9), ("-s", 0.01), ("omething", 0.2), (" entirely", 0.3))
        [scored] = ConfidenceCheck().assess("The-something entirely").sentences
        assert scored.degenerate
        assert scored.content_mean < 0.5

    def test_a_dip_on_a_grammar_word_is_not_a_guess(self, turn: TurnState) -> None:
        _generated(
            ("Your", 0.95), (" order", 0.97), (" shipped", 0.95), (" and", 0.02),
            (" arrived", 0.96), (" safely", 0.94), (".", 0.99),
        )  # fmt: skip
        [scored] = ConfidenceCheck().assess("Your order shipped and arrived safely.").sentences
        assert not scored.degenerate

    def test_the_weakest_specific_is_found(self, turn: TurnState) -> None:
        _generated(
            ("Order", 0.99), (" 10986", 0.98), (" totals", 0.95), (" $", 0.9),
            ("67", 0.05), (".27", 0.9), (".", 0.99),
        )  # fmt: skip
        [scored] = ConfidenceCheck().assess("Order 10986 totals $67.27.").sentences
        assert scored.slot_min == pytest.approx(0.05)
        assert scored.weakest_slot == "67"


def _sentence(index: int, **fields: object) -> SentenceConfidence:
    base: dict[str, object] = {
        "index": index,
        "sentence": "s",
        "mean_prob": 0.9,
        "min_prob": 0.9,
        "weakest": "w",
        "low": False,
    }
    return SentenceConfidence.model_validate({**base, **fields})


def _assessment(*verdicts: DetectionVerdict) -> GenerationAssessment:
    return GenerationAssessment(verdicts=list(verdicts))


class TestTheGate:
    def test_a_degenerate_sentence_with_no_claim_gets_a_verdict_of_its_own(self) -> None:
        confidence = ConfidenceAssessment(
            available=True, model="m", sentences=[_sentence(0, degenerate=True, content_mean=0.3)]
        )
        [verdict] = gate(_assessment(), confidence).verdicts
        assert verdict.reason_code == LOW_CONFIDENCE_DEGENERATE
        assert verdict.sentence == 0 and not verdict.supported and not verdict.undetermined

    def test_an_unconfirmed_shaky_specific_is_cut(self) -> None:
        undecided = DetectionVerdict(claim="c", supported=False, confidence=0, undetermined=True)
        confidence = ConfidenceAssessment(
            available=True, model="m", sentences=[_sentence(0, slot_min=0.05, weakest_slot="67")]
        )
        [verdict] = gate(
            _assessment(undecided.model_copy(update={"sentence": 0})), confidence
        ).verdicts
        assert verdict.reason_code == LOW_CONFIDENCE_SPECIFIC

    @pytest.mark.parametrize(
        "verdict",
        [
            DetectionVerdict(claim="c", supported=True, confidence=1, sentence=0),
            DetectionVerdict(claim="c", supported=False, confidence=0.9, sentence=0),
        ],
        ids=["confirmed by the record", "contradicted keeps its own reason"],
    )
    def test_a_decided_claim_keeps_its_verdict(self, verdict: DetectionVerdict) -> None:
        confidence = ConfidenceAssessment(
            available=True, model="m", sentences=[_sentence(0, slot_min=0.05, weakest_slot="67")]
        )
        assert gate(_assessment(verdict), confidence).verdicts == [verdict]


class TestTheGuardCutsWhatWasGuessed:
    async def test_the_reported_garble_is_cut_and_says_why(self, turn: TurnState) -> None:
        """Reported: TRIMMED · removed 0 of 1 · "a claim is contradicted" — nothing was."""
        _generated(("The", 0.9), ("-s", 0.01), ("omething", 0.2), (" entirely", 0.3))
        guard = ResponseGuardrail(DetectionLadder(create_scoring_cascade()))
        result = await guard.enforce("The-something entirely", [])
        assert result.escalated and result.withheld
        assert result.reason_code == LOW_CONFIDENCE_DEGENERATE
        assert result.removed_sentences == ["The-something entirely"]
