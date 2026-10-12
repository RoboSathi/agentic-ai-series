"""The ops agent's hallucination check: figures against rows, then the platform's gate.

Token confidence is scripted (a `ConfidenceCheck` returning a fixed assessment); the
figures check and the gate are the real ones.
"""

from typing import Any

from agent_platform.config import get_settings
from agent_platform.hallucination_check.confidence import (
    ConfidenceAssessment,
    ConfidenceCheck,
    SentenceConfidence,
)
from ops_agent.config import get_agent_settings
from ops_agent.domain import QueryRecord
from ops_agent.guardrails import FigureCheck, OpsGrounding

QUERY = QueryRecord(
    sql="SELECT n FROM t", columns=["kind", "n"], rows=[["allow", 1234]], row_count=1
)
ANSWER = "There were 1234 allows. There were 4217 denies."


class Scripted(ConfidenceCheck):
    """Level 1 with the sentences it was handed, instead of the turn's logprobs."""

    def __init__(self, *sentences: SentenceConfidence) -> None:
        super().__init__()
        self._sentences = list(sentences)

    @property
    def enabled(self) -> bool:
        return True

    def assess(self, text: str) -> ConfidenceAssessment:
        return ConfidenceAssessment(available=True, model="m", sentences=self._sentences)


def _sentence(index: int, text: str, slot: float | None, mean: float = 0.95) -> SentenceConfidence:
    return SentenceConfidence(
        index=index, sentence=text, mean_prob=mean, min_prob=slot or mean, weakest="x",
        low=False, content_mean=mean, slot_min=slot, weakest_slot="4217" if slot else "",
    )  # fmt: skip


def _grounding(*sentences: SentenceConfidence) -> OpsGrounding:
    agent = get_agent_settings()
    confidence = Scripted(*sentences) if sentences else None
    return OpsGrounding(FigureCheck(agent), agent, confidence=confidence)


def _shaky() -> float:
    return get_settings().hallucination_check.confidence.slot_min_prob / 2


def test_a_confident_figure_no_row_holds_is_kept_and_named_unverified(ops_config: None) -> None:
    checked = _grounding(
        _sentence(0, "There were 1234 allows.", 0.99), _sentence(1, "x", 0.99)
    ).check(ANSWER, [QUERY])
    assert checked.response == ANSWER
    assert checked.unverified == ["4217"]
    assert checked.status == "warning" and checked.card.outcome.startswith("FLAGGED")
    assert checked.card.tally == "1 supported · 1 unverified of 2 claims"


def test_a_guessed_figure_no_row_confirms_is_cut(ops_config: None) -> None:
    checked = _grounding(
        _sentence(0, "There were 1234 allows.", 0.99),
        _sentence(1, "There were 4217 denies.", _shaky()),
    ).check(ANSWER, [QUERY])
    assert checked.response == "There were 1234 allows."
    assert checked.escalated and checked.status == "warning"
    assert checked.card.outcome.startswith("TRIMMED · removed 1 of 2")
    assert checked.card.removed_sentences == ["There were 4217 denies."]
    verdicts = {claim.claim: claim.verdict for claim in checked.card.claim_checks}
    assert verdicts == {"1234": "supported", "4217": "low_confidence"}


def test_a_guessed_figure_a_row_holds_is_confirmed(ops_config: None) -> None:
    checked = _grounding(_sentence(0, "There were 1234 allows.", _shaky())).check(
        "There were 1234 allows.", [QUERY]
    )
    assert not checked.escalated and checked.status == "completed"
    assert checked.card.outcome.startswith("PASSED")


def test_when_every_sentence_is_cut_the_answer_is_withheld(ops_config: None) -> None:
    checked = _grounding(_sentence(0, "There were 4217 denies.", _shaky())).check(
        "There were 4217 denies.", [QUERY]
    )
    assert checked.status == "blocked"
    assert checked.response == get_agent_settings().words.withheld


def test_the_card_is_the_support_agents_shape_levels_first(ops_config: None) -> None:
    card = (
        _grounding(_sentence(0, "There were 1234 allows.", 0.99))
        .check("There were 1234 allows.", [QUERY])
        .card
    )
    assert [level.level for level in card.levels] == [1, 2]
    assert card.confidence["available"] is True
    assert card.rule == get_agent_settings().cards.rule
    [claim] = card.claim_checks
    [step] = claim.readings
    assert step["evidence"][0]["where"] == "Query 1 · row 1"
    assert step["evidence"][0]["excerpt"] == "kind=allow, n=1234"


def test_without_token_confidence_only_the_figures_decide(ops_config: None) -> None:
    checked = _grounding().check(ANSWER, [QUERY])
    assert checked.response == ANSWER and checked.card.confidence == {}
    assert [level.level for level in checked.card.levels] == [2]


async def test_the_turns_card_is_the_platforms_hallucination_check(harness: Any) -> None:
    from tests.ops_agent.conftest import query_call

    run = harness([{"tool_calls": query_call()}, {"content": "There were 12 allows."}])
    await run.agent.handle("How many allows?", "s1")
    [card] = [event for event in run.feed.events if event.stage == "hallucination_check"]
    assert {"outcome", "tally", "levels", "claim_checks", "rule"} <= card.fields.keys()
