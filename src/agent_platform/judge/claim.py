"""Tier 5 of the scoring cascade: a claim, judged by the platform judge model.

The cascade reaches this tier only when every cheaper one declined: no literal match, no
rule, no confident similarity, no confident entailment. The judge says supported,
contradicted or unsure, with a confidence; the cascade trusts it only at or above
`hallucination_check.scoring.judge_confidence_threshold`, and passes anything below on to a
person. Unsure, and a judge that could not answer, are verdicts of no confidence, so they
always pass on: the judge never turns an outage into an answer.
"""

from functools import cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from agent_platform.config import get_settings
from agent_platform.config.settings import JudgeSectionSettings
from agent_platform.judge.base import JudgeCall, ModelJudge
from agent_platform.llm.client import LLMClient
from agent_platform.llm.types import Message
from agent_platform.scoring.types import Claim, ScorerTier, Verdict

_PROMPT = Path(__file__).resolve().parent / "prompts" / "claim_judge.md"
_TIER: ScorerTier = "calibrated_llm_judge"
# The judge's word for "cannot settle it", and the cascade's.
_UNSURE = "unsure"


class ClaimAnswer(BaseModel):
    """The judge's answer about one claim."""

    reason: str = ""
    label: Literal["supported", "contradicted", "unsure"]
    confidence: float = Field(ge=0.0, le=1.0)


class ClaimJudge(ModelJudge[Claim, ClaimAnswer, Verdict]):
    """The cascade's calibrated judge (`scoring.CalibratedJudge`), on the platform judge."""

    def __init__(
        self, client: LLMClient | None = None, settings: JudgeSectionSettings | None = None
    ) -> None:
        super().__init__(ClaimAnswer, client, settings)

    def render(self, request: Claim) -> list[Message]:
        content = claim_prompt().format(
            claim=request.text, evidence=self._evidence(request.evidence)
        )
        return [Message(role="user", content=content)]

    def verdict(self, request: Claim, answer: ClaimAnswer, call: JudgeCall) -> Verdict:
        if answer.label == _UNSURE:
            return self._unresolved(answer.reason, call.cost_usd)
        return Verdict(
            label="supported" if answer.label == "supported" else "contradicted",
            score=answer.confidence,
            tier=_TIER,
            rationale=answer.reason,
            cost_usd=call.cost_usd,
        )

    def abstain(self, request: Claim, reason: str) -> Verdict:
        return self._unresolved(reason)

    def _evidence(self, evidence: list[str]) -> str:
        """The evidence, one item a line, capped at `judge.max_evidence_chars` (0: all)."""
        joined = "\n".join(f"- {item}" for item in evidence)
        limit = self._settings.max_evidence_chars
        return joined[:limit] if limit else joined

    @staticmethod
    def _unresolved(reason: str, cost_usd: float = 0.0) -> Verdict:
        return Verdict(
            label="unresolved", score=0.0, tier=_TIER, rationale=reason, cost_usd=cost_usd
        )


@cache
def claim_prompt() -> str:
    """The claim judge's question (`prompts/claim_judge.md`)."""
    return _PROMPT.read_text(encoding="utf-8").strip()


def create_claim_judge(client: LLMClient | None = None) -> ClaimJudge | None:
    """The cascade's judge, or None when its tier is switched off (no judge is built)."""
    if not get_settings().hallucination_check.scoring.tiers.calibrated_llm_judge:
        return None
    return ClaimJudge(client)


__all__ = ["ClaimAnswer", "ClaimJudge", "claim_prompt", "create_claim_judge"]
