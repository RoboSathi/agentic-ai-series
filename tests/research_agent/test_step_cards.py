"""REFLECT and GAP FILL are step cards: one card per step, naming the criteria, the model's part
and why each follow-up was kept or dropped."""

import json
from typing import Any

import pytest

from agent_platform.guardrails.guards import PydanticValidator
from agent_platform.llm.client import LLMClient
from agent_platform.observability import StageReporter
from agent_platform.observability.status_feed import InMemoryLiveStatusFeed
from agent_platform.observability.types import StatusEvent
from agent_platform.verification.acceptance import CheckVerdict
from research_agent.domain.plan import SubQuestion
from research_agent.domain.rubric import Rubric, RubricCheck, RubricItem
from research_agent.llm.structured import LLM_STAGE, StructuredCaller
from research_agent.prompts.base import RolePrompt
from research_agent.prompts.reflector import ReflectorAnswer
from research_agent.research.cards import FollowUpStep, gap_fill_card, reflect_card
from research_agent.research.reflector import TRIED, FollowUpProposal, Rejected
from tests.conftest import ScriptedLLM

pytestmark = pytest.mark.usefixtures("ra_config")


class _Roles:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def client_for(self, role: str) -> LLMClient:
        return self._llm

    def model_key(self, role: str) -> str:
        return f"scripted:{role}"


def _gap() -> CheckVerdict:
    return CheckVerdict(
        item_id="R7",
        kind="deterministic",
        name="coverage",
        passed=False,
        score=0.0,
        threshold=1.0,
        must_pass=True,
        detail="no source dated after 2024",
    )


def _rubric() -> Rubric:
    item = RubricItem(
        item_id="R7",
        text="Name two hiring trends since 2024",
        check=RubricCheck(kind="deterministic", name="coverage"),
        threshold=1.0,
        must_pass=True,
        origin="template",
    )
    return Rubric(items=(item,))


def _step(planned: list[SubQuestion], proposal: FollowUpProposal) -> FollowUpStep:
    return FollowUpStep(
        gaps=[_gap()],
        proposal=proposal,
        planned=planned,
        reason="",
        rubric=_rubric(),
        entries=[],
        limit=200,
    )


def _text(sections: dict[str, Any]) -> str:
    return "\n".join(
        str(line.get("text") if isinstance(line, dict) else line)
        for lines in sections.values()
        for line in lines
    )


_NODE = SubQuestion(
    node_id="Q5",
    question="Which AI hiring trends were reported in 2025?",
    rubric_items=("R7",),
    queries=("AI jobs report 2025",),
)


class TestTheReflectCard:
    def test_it_names_the_criterion_and_what_is_missing_not_just_its_id(self) -> None:
        _, sections = reflect_card(1, True, _step([_NODE], FollowUpProposal(accepted=[_NODE])))
        text = _text(sections)
        assert "Name two hiring trends since 2024" in text
        assert "no source dated after 2024" in text

    def test_a_dropped_follow_up_says_why(self) -> None:
        proposal = FollowUpProposal(rejected=[Rejected(question="Old query again?", why=TRIED)])
        _, sections = reflect_card(1, False, _step([], proposal))
        assert "Old query again?" in _text(sections)

    def test_gap_fill_has_the_same_shape(self) -> None:
        heading, sections = gap_fill_card(1, _step([_NODE], FollowUpProposal(accepted=[_NODE])))
        assert "R7" in heading
        assert list(sections) == list(
            reflect_card(1, True, _step([_NODE], FollowUpProposal(accepted=[_NODE])))[1]
        )


class TestATracedCallDrawsNoCardOfItsOwn:
    async def test_the_step_gets_the_call_and_the_feed_gets_no_llm_card(self) -> None:
        feed = InMemoryLiveStatusFeed(max_events=0)
        events: list[StatusEvent] = []
        feed.subscribe(events.append)
        answer = json.dumps({"follow_ups": [], "rationale": "R7 needs 2025 data"})
        caller = StructuredCaller(
            _Roles(ScriptedLLM([{"content": answer}])),
            PydanticValidator(),
            StageReporter(feed, run_id="t"),
        )
        prompt = RolePrompt(role="reflector", instructions="Propose follow-ups.", payload={})
        parsed, call = await caller.call_traced(prompt, ReflectorAnswer)
        assert parsed.rationale == "R7 needs 2025 data"
        assert call.reasoning == "R7 needs 2025 data" and call.tokens > 0
        assert not [e for e in events if e.stage == LLM_STAGE and e.status == "completed"]


class TestAReflectionTheStopRuleDecided:
    """No model was asked: the card still says what decided, on what, and each criterion."""

    def _sections(self) -> dict[str, Any]:
        met = _gap().model_copy(
            update={"passed": True, "detail": "3 domains", "evidence": ["a.com"]}
        )
        step = FollowUpStep(
            gaps=[],
            proposal=FollowUpProposal(),
            planned=[],
            reason="the question is answered with cited findings",
            rubric=_rubric(),
            entries=[],
            limit=200,
            verdicts=[met],
            rule="question_answered",
            archetype="question_answering",
            entities=(4, 7),
        )
        return reflect_card(1, False, step)[1]

    def test_it_names_the_rule_that_decided_and_its_inputs(self) -> None:
        text = _text(self._sections())
        assert "decided in code by the question answered stop rule" in text
        assert "the question is answered with cited findings" in text
        assert "1 of 1 criteria met" in text and "entities 4 → 7" in text

    def test_every_criterion_is_listed_with_its_result_and_evidence(self) -> None:
        text = _text(self._sections())
        assert "✓ R7 · Name two hiring trends since 2024" in text
        assert "R7 · coverage: a.com" in text


# -- round 2: plan, synthesize, report check, audit -----------------------------------------

_CALL = __import__("agent_platform.observability.step_card", fromlist=["ModelCall"]).ModelCall(
    role="planner", model="m", tokens=100, seconds=1.0, response='{"nodes": []}'
)


def _plan_contract() -> Any:
    from research_agent.config import get_agent_settings
    from research_agent.domain.rubric import Contract
    from research_agent.domain.scope import ScopeBrief

    brief = ScopeBrief(
        archetype="explainer",
        topic="agent memory",
        budget=get_agent_settings().budget,
        concepts=("x",),
    )
    return Contract.freeze("c", brief, _rubric(), brief.budget)


class TestThePlanCard:
    def test_a_repaired_plan_says_what_was_wrong_and_names_its_model(self) -> None:
        from research_agent.domain.plan import PlanDag
        from research_agent.research.cards import plan_card

        plan = PlanDag(nodes=(_NODE,), rationale="cover R7")
        two = _CALL.model_copy(update={"calls": 2})
        heading, sections = plan_card(_plan_contract(), plan, 1, two, repaired_problems=["cycle"])
        text = _text(sections)
        assert "agent memory" in text and "R7 · Name two hiring trends since 2024" in text
        assert "PLANNER · m · 100 tokens · 1.0s · 2 calls" in text
        assert "the first plan was repaired: cycle" in text

    def test_a_rejected_plan_lists_its_problems(self) -> None:
        from research_agent.research.cards import plan_card

        heading, sections = plan_card(
            _plan_contract(), None, 0, _CALL, rejected=["no node serves R7"]
        )
        assert "no node serves R7" in heading and "no node serves R7" in _text(sections)


def _entry(entry_id: str, url: str, span: str) -> Any:
    from agent_platform.evidence.types import EvidenceEntry

    return EvidenceEntry(entry_id=entry_id, claim=span, span=span, source_url=url, source_hash="h")


class TestTheReportCheckNamesTheSourceOfEverySentence:
    def _sections(self) -> dict[str, Any]:
        from agent_platform.evidence.types import LedgerSnapshot
        from research_agent.research.cards import guard_card
        from research_agent.synthesis.citation_guard import GuardedReport, SentenceCheck

        ledger = LedgerSnapshot(entries=(_entry("E3", "https://www.wef.org/r", "Hiring rose 4%."),))
        guarded = GuardedReport(
            report=__import__("research_agent.domain.report", fromlist=["Report"]).Report(
                title="t", status="PARTIAL"
            ),
            checked=2,
            supported=1,
            dropped=["Hiring doubled [E3]."],
            checks=[
                SentenceCheck(sentence="Hiring rose [E3].", cited=["E3"]),
                SentenceCheck(sentence="Hiring doubled [E3].", outcome="dropped", cited=["E3"]),
            ],
        )
        return guard_card(guarded, ledger, "nli-model", 200)[1]

    def test_a_supported_sentence_names_its_entry_and_host(self) -> None:
        text = _text(self._sections())
        assert "✓ Hiring rose [E3]. [SUPPORTED]" in text
        assert 'supported by [E3 · wef.org]: "Hiring rose 4%."' in text

    def test_a_removed_sentence_says_why(self) -> None:
        text = _text(self._sections())
        assert "✗ Hiring doubled [E3]. [DROPPED]" in text and "removed: not supported" in text


class TestTheAuditCard:
    def test_the_judges_reasons_are_its_reasoning_and_its_calls_are_counted(self) -> None:
        from research_agent.audit.verdict import decide
        from research_agent.research.cards import JudgeSetup, audit_card

        verdict = _gap().model_copy(
            update={"kind": "judge", "passed": True, "score": 1.0, "detail": "two trends named"}
        )
        result = decide([verdict], _rubric(), 0.8, 0.5)
        setup = JudgeSetup(
            judge_model="judge-m",
            usage={"calls": 3, "tokens": 900, "seconds": 4.5},
            judge_role="judge",
        )
        _, sections = audit_card([verdict], _rubric(), result, 0.8, 0.5, setup, 200)
        text = _text(sections)
        assert "JUDGE · judge-m · 900 tokens · 4.5s · 3 calls" in text
        assert "R7: two trends named" in text
