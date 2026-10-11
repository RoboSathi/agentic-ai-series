"""REFLECT's critique: once every check passes, the model judges whether the claims answer
the question — the checks count citations and terms, not meaning.

Regression for a run that stopped with no model call at all: "impact of mirror neurons on
learning" met every criterion by count, and nothing asked whether it was answered.
"""

from types import SimpleNamespace
from typing import Any

import pytest

from research_agent.config import get_agent_settings
from research_agent.domain.rubric import Contract
from research_agent.domain.scope import ScopeBrief
from research_agent.prompts import reflector as prompt
from research_agent.research.reflector import TRIED, Reflector
from research_agent.research.stop_rules import stop_rule_for
from research_agent.rubric.generator import RubricGenerator

pytestmark = pytest.mark.usefixtures("ra_config")


def _contract() -> Contract:
    brief = ScopeBrief(
        archetype="general", topic="impact of mirror neurons on learning",
        budget=get_agent_settings().budget,
    )  # fmt: skip
    rubric = RubricGenerator(get_agent_settings().rubric.resolved_templates_path).generate(brief)
    return Contract.freeze("c1", brief, rubric, brief.budget)


class _Caller:
    def __init__(self, answer: prompt.CritiqueAnswer) -> None:
        self._answer = answer
        self.prompts: list[Any] = []

    async def call_traced(self, request: Any, schema: Any) -> tuple[Any, None]:
        self.prompts.append(request)
        return self._answer, None


def _ctx(contract: Contract) -> Any:
    entries = [SimpleNamespace(claim="Mirror neurons fire when an action is observed.")]
    return SimpleNamespace(contract=contract, ledger=SimpleNamespace(entries=entries))


def _reflector(answer: prompt.CritiqueAnswer) -> tuple[Reflector, _Caller]:
    caller = _Caller(answer)
    return Reflector(caller, stop_rule_for, max_follow_ups=2), caller  # type: ignore[arg-type]


async def test_a_sufficient_answer_proposes_nothing_and_says_why() -> None:
    critique = "The claims explain the mechanism and cite two studies on imitation learning."
    reflector, caller = _reflector(prompt.CritiqueAnswer(sufficient=True, critique=critique))
    proposal = await reflector.critique(_ctx(_contract()), [], [], 2)
    assert proposal.sufficient and proposal.reasoning == critique and not proposal.accepted
    assert caller.prompts[0].instructions == prompt.CRITIQUE_INSTRUCTIONS


def _judged(contract: Contract) -> str:
    return next(i.item_id for i in contract.rubric.items if i.check.kind == "judge")


async def test_an_insufficient_answer_proposes_follow_ups_filtered_in_code() -> None:
    contract = _contract()
    # Every check passed: only a judge-scored criterion ("answers the question") is a target.
    item = _judged(contract)
    answer = prompt.CritiqueAnswer(
        sufficient=False,
        critique="Nothing on learning outcomes in classrooms.",
        follow_ups=[
            prompt.FollowUp(
                question="What do studies find on classroom learning?",
                rubric_items=[item],
                queries=["mirror neurons classroom learning studies"],
            ),
            prompt.FollowUp(question="Repeat", rubric_items=[item], queries=["already tried"]),
        ],
    )
    reflector, _ = _reflector(answer)
    proposal = await reflector.critique(_ctx(contract), [], ["already tried"], 2)
    assert proposal.sufficient is False
    assert proposal.reasoning == "Nothing on learning outcomes in classrooms."
    assert [f.question for f in proposal.accepted] == [
        "What do studies find on classroom learning?"
    ]
    assert [r.why for r in proposal.rejected] == [TRIED]


async def test_a_follow_up_for_a_criterion_that_already_passed_is_dropped() -> None:
    contract = _contract()
    passed = next(i.item_id for i in contract.rubric.items if i.check.kind != "judge")
    answer = prompt.CritiqueAnswer(
        sufficient=False,
        critique="Thin.",
        follow_ups=[prompt.FollowUp(question="More citations?", rubric_items=[passed],
                                    queries=["mirror neurons citations"])],
    )  # fmt: skip
    reflector, _ = _reflector(answer)
    proposal = await reflector.critique(_ctx(contract), [], [], 2)
    assert proposal.accepted == [] and len(proposal.rejected) == 1


async def test_without_a_model_there_is_no_critique() -> None:
    reflector = Reflector(None, stop_rule_for, max_follow_ups=2)
    proposal = await reflector.critique(_ctx(_contract()), [], [], 2)
    assert proposal.sufficient is None and proposal.call is None and not proposal.reasoning


def test_the_critique_payload_holds_the_whole_rubric_and_no_page_text() -> None:
    contract = _contract()
    checks = [{"item_id": contract.rubric.ids[0], "passed": False, "detail": "0/1"}]
    request = prompt.build_critique(contract, checks, ["q"], ["a claim"], 2)
    assert [item["item_id"] for item in request.payload["rubric"]] == contract.rubric.ids
    assert request.payload["claims_so_far"] == ["a claim"]
    assert request.payload["checks"] == checks


def _step(proposal: Any, contract: Contract) -> Any:
    from research_agent.research import cards

    return cards.FollowUpStep(
        gaps=[], proposal=proposal, planned=[], reason="stop", rubric=contract.rubric,
        entries=[], limit=200, verdicts=[], rule="question_answered", archetype="general",
    )  # fmt: skip


def test_the_card_shows_the_model_its_reasoning_and_its_verdict() -> None:
    from agent_platform.observability.step_card import ModelCall
    from research_agent.research import cards
    from research_agent.research.reflector import FollowUpProposal

    call = ModelCall(role="reflector", model="m", tokens=10, seconds=1.0)
    proposal = FollowUpProposal(call=call, reasoning="Covers the mechanism.", sufficient=True)
    _, sections = cards.reflect_card(1, False, _step(proposal, _contract()))
    text = str(sections)
    assert "REFLECTOR" in text and "Covers the mechanism." in text
    assert "decided in code" not in text.casefold()


def test_without_a_model_call_the_card_says_it_was_decided_in_code() -> None:
    from research_agent.research import cards
    from research_agent.research.reflector import FollowUpProposal

    _, sections = cards.reflect_card(1, False, _step(FollowUpProposal(), _contract()))
    assert "REFLECTOR" not in str(sections)
