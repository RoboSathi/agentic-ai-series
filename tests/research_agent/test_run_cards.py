"""The run's cards say what happened once: no limit card for a plan's own rounds, and a judge
card that names who judged and how the score was reached."""

import pytest

from agent_platform.loop_control import BudgetGuard
from agent_platform.verification.acceptance import CheckVerdict
from research_agent.audit.verdict import decide
from research_agent.config import get_agent_settings
from research_agent.domain.rubric import Rubric, RubricCheck, RubricItem
from research_agent.graph.builder import _announce_cap
from research_agent.research.cards import JudgeSetup, audit_card

pytestmark = pytest.mark.usefixtures("ra_config")


class _Telemetry:
    def __init__(self) -> None:
        self.cards: list[str] = []

    def card(self, stage: str, heading: str, *_: object, **__: object) -> None:
        self.cards.append(heading)


def _guard(telemetry: _Telemetry) -> BudgetGuard:
    agent = get_agent_settings()
    return BudgetGuard(
        {"rounds": 1, "fetches": 1},
        on_exhausted=lambda report: _announce_cap(telemetry, report, agent),  # type: ignore[arg-type]
    )


def test_using_up_the_planned_rounds_draws_no_limit_card() -> None:
    telemetry = _Telemetry()
    _guard(telemetry).charge("rounds", 1)
    assert telemetry.cards == []


def test_a_refused_page_read_still_does() -> None:
    telemetry = _Telemetry()
    guard = _guard(telemetry)
    guard.charge("fetches", 1)
    assert len(telemetry.cards) == 1 and "fetches" in telemetry.cards[0]


def _item(item_id: str, kind: str, must: bool, weight: float) -> RubricItem:
    return RubricItem(
        item_id=item_id,
        text=f"criterion {item_id}",
        check=RubricCheck(kind=kind, name="x"),  # type: ignore[arg-type]
        threshold=1.0,
        must_pass=must,
        weight=weight,
        origin="template",
    )


def test_the_judge_card_names_its_models_in_the_body_and_works_the_score_through() -> None:
    rubric = Rubric(items=(_item("R1", "nli", True, 3), _item("R2", "judge", False, 1)))
    verdicts = [
        CheckVerdict(
            item_id="R1",
            kind="nli",
            name="x",
            passed=False,
            score=0.5,
            threshold=1.0,
            must_pass=True,
        ),
        CheckVerdict(
            item_id="R2",
            kind="judge",
            name="x",
            passed=True,
            score=1.0,
            threshold=0.7,
            must_pass=False,
        ),
    ]
    result = decide(verdicts, rubric, 0.8, 0.5)
    setup = JudgeSetup(
        judge_model="nim:ultra",
        writer_model="nim:super",
        nli_model="deberta",
        usage={"calls": 1, "tokens": 900, "seconds": 5.3},
    )
    heading, sections = audit_card(verdicts, rubric, result, 0.8, 0.5, setup, 110)
    # The heading never names a model (config may swap it); the body names every one.
    assert heading.startswith("1/2 criteria met · score 62% · PARTIAL")
    assert "nim:ultra" not in heading
    text = str(sections)
    assert "nim:ultra" in text and "nim:super" in text and "deberta" in text and "5.3" in text
    assert "(3×0.50 + 1×1.00) ÷ 4 = 62%" in text
