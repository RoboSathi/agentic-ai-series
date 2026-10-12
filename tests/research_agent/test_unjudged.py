"""When the judge (and every backup) fails, the criterion is unverified — never a crashed run."""

from datetime import date
from typing import Any

import pytest

from agent_platform.evidence import LedgerSnapshot
from agent_platform.verification.acceptance import CheckSpec, CheckVerdict
from research_agent.audit.context import JudgeItem
from research_agent.audit.independent import LlmJudge, _result
from research_agent.domain.report import Report

pytestmark = pytest.mark.usefixtures("ra_config")

SPEC = CheckSpec(item_id="R3", kind="judge", name="explains_tradeoffs", threshold=0.6)
REPORT = Report(title="t", status="PARTIAL")


ITEM = JudgeItem(SPEC, REPORT, LedgerSnapshot(entries=()), date(2026, 10, 4))


class _DownCaller:
    async def call_traced(self, prompt: Any, schema: Any) -> Any:
        raise TimeoutError("the judge did not answer")


class _Answer:
    score, rationale, evidence = 0.8, "explains both", ["E1"]


class _Call:
    model, backup = "ultra", ""


class _Caller:
    async def call_traced(self, prompt: Any, schema: Any) -> Any:
        return _Answer(), _Call()


async def test_a_failed_judge_returns_an_unscored_outcome_not_an_exception() -> None:
    outcome = await LlmJudge(_DownCaller()).judge(ITEM)  # type: ignore[arg-type]
    assert outcome.applicable is False and outcome.score == 0.0
    assert "did not answer" in outcome.detail


async def test_an_answer_is_the_items_score_and_names_the_model() -> None:
    """The research agent's judge is the platform judge's skeleton, on its own router."""
    from agent_platform.judge import LlmJudge as PlatformLlmJudge

    judge = LlmJudge(_Caller())  # type: ignore[arg-type]
    outcome = await judge.judge(ITEM)
    assert isinstance(judge, PlatformLlmJudge)
    assert outcome.score == 0.8 and outcome.applicable is not False
    assert judge.models_used() == {"ultra"}


def test_an_unjudged_must_pass_item_is_unverified_not_a_gap_to_research() -> None:
    unjudged = CheckVerdict(
        item_id="R3", kind="judge", name="x", passed=False, score=0.0, threshold=0.6,
        must_pass=True, applicable=False,
    )  # fmt: skip
    failed = CheckVerdict(
        item_id="R4", kind="nli", name="y", passed=False, score=0.1, threshold=0.6, must_pass=True
    )
    audit = _result([unjudged, failed], REPORT, "judge", True)
    assert audit.unjudged == ["R3"]
    assert audit.must_pass_gaps == ["R4"]  # gap-fill researches R4, never the unjudged R3
