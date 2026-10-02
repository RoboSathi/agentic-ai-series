"""After a proposal, agreeing is enough: "yes" closes the interview, and approves the criteria."""

from types import SimpleNamespace

import pytest

from research_agent.config import get_agent_settings
from research_agent.domain.rubric import Rubric, RubricCheck, RubricItem
from research_agent.graph.scoping_nodes import ScopingNodes
from research_agent.graph.state import ScopingState
from research_agent.rubric.editor import render

pytestmark = pytest.mark.usefixtures("ra_config")


def _nodes() -> ScopingNodes:
    agent = get_agent_settings()
    parts = SimpleNamespace(settings=agent.scoping, commands=agent.command_names, editor=object())
    return ScopingNodes(parts)  # type: ignore[arg-type]


@pytest.mark.parametrize("reply", ["yes", "Looks good!", "ok", "go ahead"])
def test_agreeing_to_the_draft_plan_closes_the_interview(reply: str) -> None:
    state = ScopingState(phase="INTERVIEW", proposed=True, user_input=reply)
    assert _nodes().route_input(state) == "close_scoping"


def test_without_a_proposal_yes_is_just_an_answer() -> None:
    state = ScopingState(phase="INTERVIEW", proposed=False, user_input="yes")
    assert _nodes().route_input(state) == "interview_turn"


def test_a_change_after_the_proposal_refines_the_plan() -> None:
    state = ScopingState(phase="INTERVIEW", proposed=True, user_input="add policy news too")
    assert _nodes().route_input(state) == "interview_turn"


def test_agreeing_in_rubric_review_approves() -> None:
    state = ScopingState(phase="RUBRIC_REVIEW", user_input="yes")
    assert _nodes().route_input(state) == "rubric_frozen"


def test_the_rubric_reads_must_and_soft_from_config() -> None:
    messages = get_agent_settings().messages
    item = RubricItem(
        item_id="R1",
        text="Every sentence cites evidence.",
        check=RubricCheck(kind="deterministic", name="citations_per_claim"),
        threshold=1.0,
        must_pass=True,
        weight=3,
        origin="template",
    )
    text = render(Rubric(items=(item,)), messages)
    assert text.splitlines()[0] == messages.rubric_heading
    assert messages.rubric_must in text and "weight 3" in text
