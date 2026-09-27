"""Calibrated LLM judge and its calibration harness (E11, LLD §18).

A judge that has never been measured against human labels is an opinion
with a temperature setting. `JudgeCalibrator` is what turns it into an
instrument: Cohen's kappa against human adjudications, and pairwise
comparison where absolute scores are known to be unreliable.
"""

import json
import math
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from agentic_commons.config import get_settings
from agentic_commons.evals.types import Claim, Verdict
from agentic_commons.llm.client import LLMClient, create_pinned_llm_client
from agentic_commons.llm.types import CompletionRequest, Message
from agentic_commons.logging import get_logger

_PROMPT_FILE = Path(__file__).resolve().parent / "prompts" / "judge.md"

_VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["supported", "contradicted"]},
        "confidence": {"type": "number"},
        "rationale": {"type": "string"},
    },
    "required": ["label", "confidence", "rationale"],
}


@runtime_checkable
class CalibratedJudge(Protocol):
    """Adjudicates a claim, reporting calibrated confidence as the score."""

    async def judge(self, claim: Claim) -> Verdict: ...


class LlmJudge:
    """Tier-5 judge backed by an LLM, returning JSON-shaped verdicts.

    A malformed or unparseable response is reported as zero confidence
    rather than being coerced into a guess — below the configured
    threshold, so the cascade escalates instead of trusting it.
    """

    def __init__(self, client: LLMClient | None = None) -> None:
        settings = get_settings().evals.judge
        self._client = client or create_pinned_llm_client(settings.provider, settings.model)
        self._temperature = settings.temperature
        self._max_tokens = settings.max_tokens
        self._template = _PROMPT_FILE.read_text()
        self._logger = get_logger("evals")

    async def judge(self, claim: Claim) -> Verdict:
        """Ask the model to adjudicate, and trust only what parses."""
        prompt = self._template.replace("{claim}", claim.text).replace(
            "{evidence}", "\n".join(f"- {item}" for item in claim.evidence) or "(none)"
        )
        result = await self._client.complete(
            CompletionRequest(
                messages=[Message(role="user", content=prompt)],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                json_schema=_VERDICT_SCHEMA,
            )
        )
        return self._parse(result.content, result.cost_usd)

    def _parse(self, content: str, cost_usd: float) -> Verdict:
        """Parse the judge's JSON; an unreadable answer scores zero."""
        try:
            payload = json.loads(extract_json(content))
            label = payload["label"]
            confidence = float(payload["confidence"])
        except (ValueError, KeyError, TypeError) as exc:
            self._logger.warning("Judge returned an unparseable verdict", error=str(exc))
            return Verdict(
                label="unresolved",
                score=0.0,
                tier="calibrated_llm_judge",
                rationale="judge response could not be parsed",
                cost_usd=cost_usd,
            )
        return Verdict(
            label="supported" if label == "supported" else "contradicted",
            score=max(0.0, min(1.0, confidence)),
            tier="calibrated_llm_judge",
            rationale=str(payload.get("rationale", "")),
            cost_usd=cost_usd,
        )

    @classmethod
    def create_if_enabled(cls) -> "LlmJudge | None":
        """Build a judge only when both the LLM and its tier are enabled."""
        settings = get_settings()
        if not settings.llm.enabled or not settings.evals.cascade.calibrated_llm_judge:
            return None
        return cls()


def extract_json(content: str) -> str:
    """Pull the JSON object out of a response that may be fenced or prefaced."""
    start, end = content.find("{"), content.rfind("}")
    return content[start : end + 1] if start != -1 and end > start else content


class AgreementReport(BaseModel):
    """How closely the judge tracks human adjudication (E11)."""

    sample_size: int
    observed_agreement: float
    expected_agreement: float
    cohens_kappa: float
    acceptable: bool
    labels: dict[str, int] = Field(default_factory=dict)


class JudgeCalibrator:
    """Measures a judge against human labels using Cohen's kappa.

    Raw agreement flatters any judge on a skewed label distribution —
    always answering "supported" scores 90% when 90% of claims are
    supported. Kappa removes the agreement that chance alone explains,
    which is why it, not accuracy, is the gate.
    """

    def __init__(self) -> None:
        self._acceptable = get_settings().evals.kappa_acceptable

    def agreement(self, judge_labels: list[str], human_labels: list[str]) -> AgreementReport:
        """Cohen's kappa between the judge's labels and a human's."""
        if len(judge_labels) != len(human_labels):
            raise ValueError("Judge and human label sets must be the same length")
        total = len(judge_labels)
        if total == 0:
            return AgreementReport(
                sample_size=0,
                observed_agreement=0.0,
                expected_agreement=0.0,
                cohens_kappa=0.0,
                acceptable=False,
            )
        observed = sum(a == b for a, b in zip(judge_labels, human_labels)) / total
        expected = self._chance_agreement(judge_labels, human_labels)
        kappa = 1.0 if expected >= 1.0 else (observed - expected) / (1.0 - expected)
        return AgreementReport(
            sample_size=total,
            observed_agreement=observed,
            expected_agreement=expected,
            cohens_kappa=kappa,
            acceptable=kappa >= self._acceptable,
            labels=_label_counts(judge_labels),
        )

    @staticmethod
    def _chance_agreement(judge_labels: list[str], human_labels: list[str]) -> float:
        """Agreement two raters would reach at random, given their label mixes."""
        total = len(judge_labels)
        judge_counts, human_counts = _label_counts(judge_labels), _label_counts(human_labels)
        return sum(
            (judge_counts.get(label, 0) / total) * (human_counts.get(label, 0) / total)
            for label in set(judge_counts) | set(human_counts)
        )

    @staticmethod
    def pairwise_win_rate(preferences: list[str], candidate: str) -> float:
        """Share of pairwise comparisons a candidate won.

        Pairwise preference survives judges whose absolute scores drift
        between runs: "which of these two is better" stays stable when
        "rate this out of ten" does not.
        """
        if not preferences:
            return 0.0
        return preferences.count(candidate) / len(preferences)

    @staticmethod
    def standard_error(win_rate: float, comparisons: int) -> float:
        """Binomial standard error — how much of a win rate is just sample size."""
        if comparisons <= 0:
            return 0.0
        return math.sqrt(max(0.0, win_rate * (1.0 - win_rate)) / comparisons)


def _label_counts(labels: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return counts
