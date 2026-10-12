"""DeepEval, run the standard way (`evaluate()`) on the platform judge — offline, scripted."""

import json

import pytest
from pydantic import BaseModel

from agent_platform.evals.core.contracts import (
    ConversationFrame,
    GoldenCase,
    QualityMetricSpec,
    RubricBand,
    ToolCallRecord,
    Transcript,
)
from agent_platform.evals.deepeval import DeepEvalGrader, import_deepeval
from agent_platform.evals.deepeval.metrics import (
    UnknownMetricKindError,
    UnshowableParamError,
    build_metrics,
)
from agent_platform.evals.deepeval.model import judge_model
from agent_platform.evals.deepeval.test_cases import capped, conversation
from customer_support_agent.evals.plugin import CsaEvalPlugin
from tests.conftest import ScriptedLLM

# One answer that satisfies every schema these metrics ask the judge for.
ANSWER = json.dumps(
    {
        "verdict": "yes",
        "reason": "fine",
        "score": 9,
        "truths": ["it ships monday"],
        "claims": ["it ships monday"],
        "verdicts": [{"verdict": "yes", "reason": "r"}],
        "statements": ["it ships monday"],
    }
)
CASE = GoldenCase(suite="golden", case_id="c0", category="x", turns=["where?", "and now?"])
TRANSCRIPT = Transcript(
    suite="golden",
    case_id="c0",
    replies=["It ships Monday.", "Still Monday."],
    evidence=["order ships monday"],
    tool_calls=[
        ToolCallRecord(
            name="support.get_order_status", turn=0, status="completed", standing="VALID"
        )
    ],
)
FRAME = ConversationFrame(chatbot_role="support", scenario="s", expected_outcome="told when")
SPECS = [
    QualityMetricSpec(
        name="task_resolution",
        kind="conversational_geval",
        threshold=0.5,
        criteria="resolved?",
        evaluation_steps=["check it"],
        rubric=[
            RubricBand(score_range=(0, 5), expected_outcome="no"),
            RubricBand(score_range=(6, 10), expected_outcome="yes"),
        ],
    ),
    QualityMetricSpec(
        name="faithfulness",
        kind="conversational_geval",
        threshold=0.5,
        criteria="grounded?",
        evaluation_steps=["check each claim against the evidence"],
        params=["role", "content", "retrieval_context"],
    ),
    QualityMetricSpec(name="relevancy", kind="turn_relevancy", threshold=0.5),
]


class _Reason(BaseModel):
    reason: str


class TestTestCase:
    def test_each_reply_carries_its_own_tools_and_the_last_one_the_evidence(self) -> None:
        test_case = conversation(import_deepeval(), CASE, TRANSCRIPT, FRAME, evidence_cap=0)
        assert [turn.role for turn in test_case.turns] == ["user", "assistant"] * 2
        first, second = test_case.turns[1], test_case.turns[3]
        assert [tool.name for tool in first.tools_called] == ["support.get_order_status"]
        assert first.tools_called[0].output["standing"] == "VALID"
        assert second.tools_called is None
        assert first.retrieval_context is None
        assert second.retrieval_context == ["order ships monday"]
        assert test_case.expected_outcome == "told when" and test_case.chatbot_role == "support"

    def test_the_evidence_is_capped_oldest_first(self) -> None:
        assert capped(["aaaa", "bbbb", "cc"], 9) == ["aaaa", "bbbb"]
        assert capped(["aaaa"], 0) == ["aaaa"]


class TestMetrics:
    def test_one_metric_per_spec_with_the_anchored_rubric(self) -> None:
        deepeval = import_deepeval()
        metrics = build_metrics(deepeval, SPECS, judge_model(deepeval, ScriptedLLM()))
        assert metrics[0].__name__ == "task_resolution"
        assert len(metrics[0].rubric) == 2 and metrics[0].evaluation_steps == ["check it"]

    def test_an_unknown_kind_is_refused(self) -> None:
        deepeval = import_deepeval()
        spec = QualityMetricSpec(name="x", kind="nonsense", threshold=0.5)
        with pytest.raises(UnknownMetricKindError):
            build_metrics(deepeval, [spec], judge_model(deepeval, ScriptedLLM()))

    def test_a_param_g_eval_cannot_show_is_refused(self) -> None:
        deepeval = import_deepeval()
        spec = QualityMetricSpec(
            name="x", kind="conversational_geval", threshold=0.5, params=["chatbot_role"]
        )
        with pytest.raises(UnshowableParamError, match="chatbot_role"):
            build_metrics(deepeval, [spec], judge_model(deepeval, ScriptedLLM()))

    def test_the_agents_metrics_build(self) -> None:
        deepeval = import_deepeval()
        specs = CsaEvalPlugin().quality_metrics()
        assert {spec.kind for spec in specs} == {"conversational_geval", "turn_relevancy"}
        model = judge_model(deepeval, ScriptedLLM())
        assert len(build_metrics(deepeval, specs, model)) == len(specs)


class TestJudgeModel:
    async def test_a_schema_answer_is_validated_and_counted(self) -> None:
        model = judge_model(import_deepeval(), ScriptedLLM([{"content": ANSWER}]))
        answer = await model.a_generate("why?", schema=_Reason)
        assert answer == _Reason(reason="fine") and model.calls == 1

    def test_the_synchronous_path_answers_too(self) -> None:
        model = judge_model(import_deepeval(), ScriptedLLM([{"content": "plain"}]))
        assert model.generate("hello") == "plain"


class TestGrader:
    async def test_every_metric_scores_through_evaluate(self) -> None:
        llm = ScriptedLLM([{"content": ANSWER}])
        grader = DeepEvalGrader(SPECS, lambda case: FRAME, client=llm)
        scores = await grader.grade(CASE, TRANSCRIPT)
        assert {score.metric for score in scores} == {
            "task_resolution",
            "faithfulness",
            "relevancy",
        }
        assert all(score.passed and score.severity == "soft" for score in scores)
        assert all(score.tier == "calibrated_llm_judge" for score in scores)
        assert sum(score.llm_calls for score in scores) >= 1

    async def test_an_unusable_judge_is_an_error_score_never_a_low_one(self) -> None:
        grader = DeepEvalGrader(
            SPECS[2:], lambda case: FRAME, client=ScriptedLLM([{"content": "no json here"}])
        )
        scores = await grader.grade(CASE, TRANSCRIPT)
        assert [score.metric for score in scores] == ["relevancy.error"]
        assert not scores[0].passed

    def test_the_version_moves_with_the_specs(self) -> None:
        first = DeepEvalGrader(SPECS, lambda case: FRAME, client=ScriptedLLM())
        second = DeepEvalGrader(SPECS[:1], lambda case: FRAME, client=ScriptedLLM())
        assert first.version != second.version
