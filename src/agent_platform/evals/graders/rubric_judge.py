"""A generic LLM judge: rubric quality and claim-level hallucination in one call.

The agent supplies the rubric (what "good" means for its suite); this
module supplies everything else — the prompt, the parsing, the scoring.
One call per case covers both jobs, because on a rate-limited account a
second call per case is the difference between a 2-minute stage and a
4-minute one.

The judge is its own model (`evals.judge.model`), not the agent's: a model
grading its own output inherits its own blind spots.
"""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from agent_platform.config import get_settings
from agent_platform.config.settings import JudgeSettings
from agent_platform.evals.contracts import (
    GoldenCase,
    GraderKind,
    Rubric,
    Score,
    Transcript,
    canonical_hash,
)
from agent_platform.evals.judge import extract_json
from agent_platform.evals.judge_client import create_judge_client
from agent_platform.llm.client import LLMClient
from agent_platform.llm.types import CompletionRequest, Message

_PROMPT = Path(__file__).resolve().parent / "prompts" / "rubric_judge.md"
_HALLUCINATION_METRIC = "hallucination_free"
_LOWEST_SCORE = 1
_NO_CLAIMS_RATE = 0.0


class _CriterionVerdict(BaseModel):
    score: int
    rationale: str = ""


class _ClaimVerdict(BaseModel):
    claim: str
    verdict: Literal["supported", "unsupported", "unverifiable"]
    evidence: str = ""


class JudgeVerdict(BaseModel):
    """What the judge returned, validated."""

    criteria: dict[str, _CriterionVerdict]
    claims: list[_ClaimVerdict] = Field(default_factory=list)

    @property
    def hallucination_rate(self) -> float:
        """Unsupported share of the checkable claims (unverifiable ones excluded)."""
        checkable = [claim for claim in self.claims if claim.verdict != "unverifiable"]
        if not checkable:
            return _NO_CLAIMS_RATE
        return sum(claim.verdict == "unsupported" for claim in checkable) / len(checkable)


class JudgeResponseError(ValueError):
    """The judge answered, but not with the JSON it was asked for."""


class RubricJudge:
    """Scores every rubric criterion and the claims' grounding for one transcript."""

    name = "rubric_judge"
    kind: GraderKind = "judge"

    def __init__(
        self,
        rubric: Rubric,
        client: LLMClient | None = None,
        settings: JudgeSettings | None = None,
    ) -> None:
        self._settings = settings or get_settings().evals.judge
        self._rubric = rubric
        self._client = client or create_judge_client(self._settings)
        self._template = _PROMPT.read_text()
        self.version = canonical_hash(
            [rubric.model_dump(), self._settings.model_dump(), self._template]
        )[:16]

    async def grade(self, case: GoldenCase, transcript: Transcript) -> list[Score]:
        result = await self._client.complete(
            CompletionRequest(
                messages=[Message(role="user", content=self.prompt(case, transcript))],
                temperature=self._settings.temperature,
                max_tokens=self._settings.max_tokens,
            )
        )
        verdict = parse_verdict(result.content)
        return self._scores(case.case_id, verdict, result.cost_usd)

    def prompt(self, case: GoldenCase, transcript: Transcript) -> str:
        """The filled template — public so a report can show what was asked."""
        fills = {
            "{scale_max}": str(self._rubric.scale_max),
            "{criteria}": _criteria_text(self._rubric),
            "{conversation}": _conversation(case, transcript),
            "{tool_calls}": _tool_calls(transcript),
            "{evidence}": _evidence(transcript, self._settings.max_evidence_chars),
        }
        text = self._template
        for token, value in fills.items():
            text = text.replace(token, value)
        return text

    def _scores(self, case_id: str, verdict: JudgeVerdict, cost: float) -> list[Score]:
        scores = [
            self._criterion_score(case_id, criterion.name, criterion.threshold, verdict)
            for criterion in self._rubric.criteria
        ]
        rate = verdict.hallucination_rate
        unsupported = [claim.claim for claim in verdict.claims if claim.verdict == "unsupported"]
        scores.append(
            self._score(
                case_id,
                _HALLUCINATION_METRIC,
                1.0 - rate,
                rate <= self._rubric.hallucination_threshold,
                1.0 - self._rubric.hallucination_threshold,
                f"{len(verdict.claims)} claims; unsupported: {unsupported}"
                if unsupported
                else f"{len(verdict.claims)} claims, none unsupported",
            )
        )
        # The call's cost and count belong to the case once, not per metric.
        scores[0] = scores[0].model_copy(update={"cost_usd": cost, "llm_calls": 1})
        return scores

    def _criterion_score(
        self, case_id: str, name: str, threshold: float, verdict: JudgeVerdict
    ) -> Score:
        given = verdict.criteria.get(name)
        if given is None:
            return self._score(case_id, name, 0.0, False, threshold, "judge did not score this")
        span = self._rubric.scale_max - _LOWEST_SCORE
        clamped = max(_LOWEST_SCORE, min(self._rubric.scale_max, given.score))
        value = (clamped - _LOWEST_SCORE) / span if span else 0.0
        return self._score(case_id, name, value, value >= threshold, threshold, given.rationale)

    def _score(
        self, case_id: str, metric: str, value: float, passed: bool, threshold: float, why: str
    ) -> Score:
        return Score(
            case_id=case_id,
            metric=metric,
            grader=self.name,
            kind=self.kind,
            severity="soft",
            value=round(value, 4),
            passed=passed,
            threshold=threshold,
            tier="calibrated_llm_judge",
            rationale=why,
        )


def parse_verdict(content: str) -> JudgeVerdict:
    """Validate the judge's JSON; anything else raises, and the grade is retried later."""
    try:
        return JudgeVerdict.model_validate(json.loads(extract_json(content)))
    except (ValueError, ValidationError) as exc:
        raise JudgeResponseError(f"judge response was not the requested JSON: {exc}") from exc


def _criteria_text(rubric: Rubric) -> str:
    return "\n".join(
        f"- {criterion.name}: {criterion.description}" for criterion in rubric.criteria
    )


def _conversation(case: GoldenCase, transcript: Transcript) -> str:
    lines: list[str] = []
    for index, turn in enumerate(case.turns):
        lines.append(f"CUSTOMER: {turn}")
        reply = transcript.replies[index] if index < len(transcript.replies) else "(no reply)"
        lines.append(f"ASSISTANT: {reply}")
    return "\n".join(lines)


def _tool_calls(transcript: Transcript) -> str:
    if not transcript.tool_calls:
        return "(none)"
    return "\n".join(
        f"- {call.name}({json.dumps(call.arguments, default=str)}) → "
        f"{call.decision or 'no ruling'} {call.reason_code} [{call.status}]"
        for call in transcript.tool_calls
    )


def _evidence(transcript: Transcript, limit: int) -> str:
    text = "\n---\n".join(transcript.evidence) or "(none)"
    return text if limit <= 0 else text[:limit]
