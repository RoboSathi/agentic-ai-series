"""LOOP CONTROL for a research run: gap-fill or finish, and why — on the card and the edge alike.

The run's routing edges cannot draw; the node before each works out the same decision and
reports it. Real nodes and resources from the composition root; the audit is built here.
"""

from typing import Any

import pytest

from agent_platform.observability.types import StatusEvent
from research_agent.audit.independent import AuditResult
from research_agent.config import get_agent_settings
from tests.research_agent.conftest import BuildHarness

_LOOP = "loop_control"


def _contract() -> Any:
    from research_agent.domain.rubric import Contract, Rubric
    from research_agent.domain.scope import ScopeBrief

    brief = ScopeBrief(
        archetype="explainer",
        topic="agent memory",
        budget=get_agent_settings().budget,
        concepts=("x",),
    )
    return Contract.freeze("c", brief, Rubric(), brief.budget)


@pytest.fixture
def bound(harness: BuildHarness) -> tuple[Any, list[StatusEvent]]:
    h = harness()
    agent = h.agent
    agent._nodes.bind(agent._resources(_contract()))
    events: list[StatusEvent] = []
    h.feed.subscribe(events.append)
    return agent._nodes, events


def _gaps(*items: str) -> AuditResult:
    return AuditResult(must_pass_gaps=list(items))


class TestTheGapFillDecision:
    def test_no_gaps_finishes(self, bound: tuple[Any, list[StatusEvent]]) -> None:
        nodes, _ = bound
        assert nodes._gap_fill_decision(_gaps()) == ("finalize", "no_gaps")

    def test_gaps_with_a_round_left_fill_them(self, bound: tuple[Any, list[StatusEvent]]) -> None:
        nodes, _ = bound
        if nodes.resources.budget.remaining("gap_fill_rounds") <= 0:
            pytest.skip("this budget allows no gap-fill round")
        assert nodes._gap_fill_decision(_gaps("R1")) == ("gap_fill", "gaps_to_fill")

    def test_gaps_with_no_round_left_say_so(self, bound: tuple[Any, list[StatusEvent]]) -> None:
        nodes, _ = bound
        budget = nodes.resources.budget
        budget.charge("gap_fill_rounds", budget.remaining("gap_fill_rounds"))
        assert nodes._gap_fill_decision(_gaps("R1"))[1] in {"gap_fill_exhausted", "budget_spent"}

    def test_the_edge_routes_on_the_same_rule(self, bound: tuple[Any, list[StatusEvent]]) -> None:
        nodes, _ = bound
        audit = _gaps("R1")
        assert nodes.after_audit({"audit": audit}) == nodes._gap_fill_decision(audit)[0]


def test_the_card_names_the_step_the_round_and_every_meter(
    bound: tuple[Any, list[StatusEvent]],
) -> None:
    nodes, events = bound
    nodes._report_loop({"round": 1}, "finalize", "no_gaps", {"must_pass_gaps": 0})
    [card] = [e for e in events if e.stage == _LOOP]
    words = get_agent_settings().run_loop
    assert card.fields["next_step"] == words.next["finalize"]
    assert card.fields["because"] == words.because["no_gaps"]
    assert {"searches", "rounds", "must_pass_gaps"} <= set(card.fields["criteria"])
