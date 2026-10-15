"""The shared LLM judge: one model chain, one skeleton (ask, validate, decide, abstain), calibration."""

from typing import Any

import pytest
from pydantic import BaseModel

from agent_platform.config import get_settings
from agent_platform.judge import (
    JudgeCalibrator,
    JudgeCall,
    ModelJudge,
    ask_validated,
    client,
    judge_rules,
)
from agent_platform.llm.types import Message
from tests.conftest import ScriptedLLM


class Grade(BaseModel):
    rationale: str
    passed: bool


class Outcome(BaseModel):
    passed: bool | None = None
    model: str = ""
    abstained: str = ""


class PassJudge(ModelJudge[str, Grade, Outcome]):
    """The smallest judge an agent could write: what is asked, and what it means."""

    def __init__(self, llm: ScriptedLLM) -> None:
        super().__init__(Grade, client=llm)

    def render(self, request: str) -> list[Message]:
        return [Message(role="user", content=f"Did this pass? {request}")]

    def verdict(self, request: str, answer: Grade, call: JudgeCall) -> Outcome:
        return Outcome(passed=answer.passed, model=call.model)

    def abstain(self, request: str, reason: str) -> Outcome:
        return Outcome(abstained=reason)


class TestClient:
    def test_the_judge_is_ultra_then_its_backups(self, monkeypatch: pytest.MonkeyPatch) -> None:
        built: list[tuple[str, str]] = []

        def pinned(provider: Any, model: str, *_: Any) -> ScriptedLLM:
            built.append((provider, model))
            return ScriptedLLM()

        monkeypatch.setattr(client, "create_pinned_llm_client", pinned)
        client.create_judge_client()
        judge = get_settings().judge
        assert built == [
            (judge.provider, judge.model),
            *((backup.provider, backup.model) for backup in judge.fallbacks),
        ]

    def test_the_judge_never_runs_on_the_agents_model(self) -> None:
        """A judge that shares the agent's model inherits its blind spots."""
        settings = get_settings()
        agent_models = {settings.llm.provider_settings().model} | {
            backup.model for backup in settings.llm.chain.fallbacks
        }
        judge_models = {settings.judge.model} | {b.model for b in settings.judge.fallbacks}
        assert settings.judge.model not in agent_models
        assert settings.llm.provider_settings().model not in judge_models


class NamedLLM(ScriptedLLM):
    @property
    def model(self) -> str:
        return "judge-model"


class TestSkeleton:
    async def test_a_valid_answer_becomes_the_callers_verdict(self) -> None:
        judge = PassJudge(NamedLLM([{"content": '{"rationale": "quoted it", "passed": true}'}]))
        assert (await judge.judge("x")).passed is True
        assert judge.models_used() == {"judge-model"} and judge.models_used() == set()

    async def test_the_shared_grading_rules_come_first(self) -> None:
        llm = ScriptedLLM([{"content": '{"rationale": "r", "passed": false}'}])
        await PassJudge(llm).judge("x")
        first = llm.requests[0].messages[0]
        assert first.role == "system" and first.content == judge_rules()

    async def test_a_fenced_answer_is_repaired_not_rejected(self) -> None:
        llm = ScriptedLLM([{"content": '```json\n{"rationale": "r", "passed": true}\n```'}])
        assert (await PassJudge(llm).judge("x")).passed is True

    async def test_an_unusable_answer_abstains_never_a_silent_fail(self) -> None:
        outcome = await PassJudge(ScriptedLLM([{"content": "probably fine"}])).judge("x")
        assert outcome.passed is None and "Grade" in outcome.abstained

    async def test_an_unreachable_judge_abstains(self) -> None:
        class Down(ScriptedLLM):
            async def complete(self, request: Any) -> Any:
                raise TimeoutError("judge down")

        outcome = await PassJudge(Down()).judge("x")
        assert outcome.passed is None and "judge down" in outcome.abstained

    async def test_a_call_reports_what_it_cost(self) -> None:
        llm = ScriptedLLM([{"content": '{"rationale": "r", "passed": true}'}])
        _, call = await ask_validated(llm, [Message(role="user", content="q")], Grade)
        assert call.calls == 1 and call.cost_usd > 0


class TestCalibration:
    def test_perfect_agreement_gives_kappa_one(self) -> None:
        report = JudgeCalibrator().agreement(["a", "b", "a"], ["a", "b", "a"])
        assert report.cohens_kappa == pytest.approx(1.0)

    def test_kappa_discounts_chance_agreement(self) -> None:
        """90% accuracy on a 90/10 split is worth nothing, and kappa says so."""
        report = JudgeCalibrator().agreement(["yes"] * 10, ["yes"] * 9 + ["no"])
        assert report.observed_agreement == 0.9
        assert report.cohens_kappa < 0.1

    def test_the_bar_is_configured(self) -> None:
        assert JudgeCalibrator(acceptable=0.0).agreement(["a"], ["a"]).acceptable

    def test_mismatched_lengths_are_rejected(self) -> None:
        with pytest.raises(ValueError):
            JudgeCalibrator().agreement(["a"], ["a", "b"])

    def test_an_empty_sample_is_not_acceptable(self) -> None:
        assert JudgeCalibrator().agreement([], []).acceptable is False

    def test_pairwise_win_rate(self) -> None:
        assert JudgeCalibrator.pairwise_win_rate(["a", "a", "b"], "a") == pytest.approx(2 / 3)

    def test_standard_error_shrinks_with_sample_size(self) -> None:
        calibrator = JudgeCalibrator()
        assert calibrator.standard_error(0.5, 1000) < calibrator.standard_error(0.5, 10)
