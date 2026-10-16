"""The scoring cascade's tier 5: a claim judged by the platform judge, trusted on its confidence.

The judge model is scripted, so every verdict is a fixed expectation. Under test: what the
judge is asked, what each answer means to the cascade, that an unsure or failed judge always
passes the claim on, and that the composition root's judge is the one the cascade consults.
"""

import json
from typing import Any

import pytest

from agent_platform.config import get_settings
from agent_platform.judge import ClaimJudge, create_claim_judge
from agent_platform.scoring import CalibratedJudge, CalibratedJudgeScorer, Claim
from agent_platform.scoring.cascade import NollmScoringCascadeImpl
from tests.conftest import ScriptedLLM

CLAIM = Claim(text="the refund was issued", evidence=["refund 41 issued on monday"])


def _answer(label: str, confidence: float) -> dict[str, Any]:
    return {"content": json.dumps({"reason": "quoted", "label": label, "confidence": confidence})}


def _judge(*script: dict[str, Any]) -> tuple[ClaimJudge, ScriptedLLM]:
    llm = ScriptedLLM(list(script))
    return ClaimJudge(llm), llm


class TestTheJudge:
    def test_it_satisfies_the_cascades_seam(self) -> None:
        assert isinstance(_judge(_answer("supported", 1.0))[0], CalibratedJudge)

    async def test_it_is_asked_about_the_claim_and_its_evidence(self) -> None:
        judge, llm = _judge(_answer("supported", 0.9))
        await judge.judge(CLAIM)
        asked = llm.requests[0].messages[-1].content
        assert CLAIM.text in asked and CLAIM.evidence[0] in asked

    async def test_a_label_is_a_verdict_scored_by_its_confidence(self) -> None:
        verdict = await _judge(_answer("contradicted", 0.85))[0].judge(CLAIM)
        assert verdict.label == "contradicted" and verdict.score == 0.85
        assert verdict.tier == "calibrated_llm_judge"

    async def test_unsure_is_unresolved_with_no_confidence(self) -> None:
        verdict = await _judge(_answer("unsure", 0.9))[0].judge(CLAIM)
        assert verdict.label == "unresolved" and verdict.score == 0.0

    async def test_an_answer_off_its_schema_abstains(self) -> None:
        verdict = await _judge({"content": "I think so"})[0].judge(CLAIM)
        assert verdict.label == "unresolved" and verdict.score == 0.0

    async def test_the_evidence_is_capped(self) -> None:
        settings = get_settings().judge.model_copy(update={"max_evidence_chars": 5})
        llm = ScriptedLLM([_answer("supported", 0.9)])
        await ClaimJudge(llm, settings).judge(CLAIM)
        assert "issued on monday" not in llm.requests[0].messages[-1].content


class TestInTheCascade:
    async def test_a_confident_verdict_resolves_the_claim(self) -> None:
        scorer = CalibratedJudgeScorer(_judge(_answer("supported", 0.95))[0])
        verdict = await scorer.resolve(CLAIM)
        assert verdict is not None and verdict.label == "supported"

    async def test_a_verdict_below_the_threshold_passes_it_on(self) -> None:
        threshold = get_settings().hallucination_check.scoring.judge_confidence_threshold
        scorer = CalibratedJudgeScorer(_judge(_answer("supported", threshold / 2))[0])
        assert await scorer.resolve(CLAIM) is None

    async def test_an_unsure_judge_passes_it_on_to_a_person(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tiers = get_settings().hallucination_check.scoring.tiers
        monkeypatch.setattr(tiers, "calibrated_llm_judge", True)
        # Only the tiers that cannot decide this claim, so it reaches the judge.
        monkeypatch.setattr(tiers, "embedding_similarity", False)
        monkeypatch.setattr(tiers, "nli_entailment", False)
        judge, llm = _judge(_answer("unsure", 1.0))
        cascade = NollmScoringCascadeImpl(judge=judge)
        verdict = await cascade.resolve(Claim(text="zzz qqq", evidence=["alpha beta"]))
        # The judge was asked (the cheap tiers declined), and its unsure went to a person.
        assert llm.requests and verdict.tier == "human_adjudication"


class TestTheFactory:
    def test_no_judge_is_built_when_its_tier_is_off(self) -> None:
        # Tests run offline (conftest), and the judge tier is a hosted model.
        assert create_claim_judge() is None

    def test_a_judge_is_built_when_its_tier_is_on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        tiers = get_settings().hallucination_check.scoring.tiers
        monkeypatch.setattr(tiers, "calibrated_llm_judge", True)
        assert isinstance(create_claim_judge(ScriptedLLM()), ClaimJudge)
