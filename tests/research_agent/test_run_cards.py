"""The run's cards say what happened once: a plan's own rounds are no intervention, a budget
stop names its mechanism, and the judge card names who judged and how the score was reached."""

import pytest

from agent_platform.loop_control import BudgetGuard
from agent_platform.runtime.verification.acceptance import CheckVerdict
from research_agent.audit.verdict import decide
from research_agent.config import get_agent_settings
from research_agent.domain.rubric import Rubric, RubricCheck, RubricItem
from research_agent.research.cards import JudgeSetup, audit_card

pytestmark = pytest.mark.usefixtures("ra_config")


def _guard() -> BudgetGuard:
    return BudgetGuard({"rounds": 1, "fetches": 1}, layers=get_agent_settings().budget_layers)


def test_a_plan_that_uses_its_one_round_and_stops_is_no_intervention() -> None:
    guard = _guard()
    guard.charge("rounds", 1)
    verdict = guard.verdict(1, 1, stopping=True)
    assert verdict.decision == "finalize"
    assert not verdict.intervened and verdict.layer is None


def test_a_budget_that_stops_the_run_names_its_own_mechanism() -> None:
    guard = _guard()
    guard.charge("rounds", 1)
    verdict = guard.verdict(1, 1, stopping=True, cause="rounds")
    assert verdict.intervened and verdict.layer == "step_count"
    guard.charge("fetches", 1)
    assert guard.verdict(1, 1, stopping=True, cause="fetches").layer == "tool_budget"


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
