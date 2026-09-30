"""The offline judge: soft quality scored by stated heuristics, no model call.

The platform's judge stage sends one prompt (its rubric template, filled)
to whatever client the plugin supplies. Under the fixture backend that
client is this one: it reads the filled template back — the criteria, the
final reply, the evidence — and answers the JSON the platform parses.
Each criterion has a heuristic a person could apply by hand, so a soft
score moves only when the report does. With `model_backend: routed` the
plugin supplies no client and the platform's configured judge model runs.
"""

import json
import re
from collections.abc import Callable

from agentic_commons.evals.contracts import Rubric
from agentic_commons.llm.client import LLMClient
from agentic_commons.llm.types import (
    CompletionRequest,
    CompletionResult,
    HealthCheckResult,
    TokenUsage,
)
from research_agent.config.settings import ReferenceJudgeSettings

_TIER, _MODEL = "fixture", "reference-judge"
_LOWEST = 1
_NOT_APPLICABLE = "not applicable to this deliverable"

Heuristic = Callable[["_Reading"], tuple[float, str]]


class _Reading:
    """The parts of one filled judge prompt this judge reads."""

    def __init__(self, prompt: str, settings: ReferenceJudgeSettings, citation: re.Pattern[str]):
        self.settings = settings
        self.citation = citation
        self.reply = _between(prompt, settings.conversation_label, settings.tool_calls_label)
        evidence = prompt.partition(settings.evidence_label)[2].strip()
        self.spans = [_span(item) for item in evidence.split(settings.evidence_separator) if item]

    @property
    def lines(self) -> list[str]:
        return [line.strip() for line in self.reply.splitlines() if line.strip()]

    @property
    def cited(self) -> list[str]:
        return [line for line in self.lines if self.citation.search(line)]

    def grounded(self, line: str) -> bool:
        claim = self.citation.sub("", line).strip(" .[]")
        return any(span and span in claim for span in self.spans)


def _between(text: str, start: str, end: str) -> str:
    return text.partition(start)[2].partition(end)[0]


def _span(item: str) -> str:
    """`[E3] the span (https://…)` → `the span` (without its final full stop)."""
    body = item.strip().partition("] ")[2].rpartition(" (")[0]
    return body.rstrip(".")


def _tradeoffs(r: _Reading) -> tuple[float, str]:
    if not any(line.startswith("#") and r.settings.decision_heading in line for line in r.lines):
        return 1.0, _NOT_APPLICABLE
    markers = r.settings.tradeoff_markers
    stated = all(any(line.startswith(m) for line in r.lines) for m in markers)
    return (
        (1.0, "trade-offs and their counterweights are stated")
        if stated
        else (
            0.0,
            "no trade-off is stated against its counterweight",
        )
    )


def _grounding(r: _Reading) -> tuple[float, str]:
    cited = r.cited
    if not cited:
        return 0.0, "no cited sentence"
    grounded = sum(r.grounded(line) for line in cited)
    return grounded / len(cited), f"{grounded}/{len(cited)} cited sentences quote their evidence"


def _honest_status(r: _Reading) -> tuple[float, str]:
    partial = f"{r.settings.status_prefix}{r.settings.partial_status}" in r.reply
    if not partial:
        return 1.0, "complete delivery"
    declared = any(r.settings.limits_heading in line for line in r.lines)
    return (
        (1.0, "partial delivery declares its limits")
        if declared
        else (
            0.0,
            "partial delivery hides its limits",
        )
    )


HEURISTICS: dict[str, Heuristic] = {
    "tradeoffs_fair": _tradeoffs,
    "grounded_explanation": _grounding,
    "honest_status": _honest_status,
}


class ReferenceJudge(LLMClient):
    """Answers the platform's rubric-judge prompt from the reply and its evidence."""

    def __init__(
        self, rubric: Rubric, settings: ReferenceJudgeSettings, citation_pattern: str
    ) -> None:
        self._rubric = rubric
        self._settings = settings
        self._citation = re.compile(citation_pattern)

    @property
    def tier(self) -> str:
        return _TIER

    @property
    def model(self) -> str:
        return _MODEL

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        reading = _Reading(request.messages[-1].content, self._settings, self._citation)
        verdict = {"criteria": self._criteria(reading), "claims": _claims(reading)}
        return CompletionResult(
            content=json.dumps(verdict),
            model=_MODEL,
            finish_reason="stop",
            tier=_TIER,
            usage=TokenUsage(),
        )

    async def health_check(self) -> HealthCheckResult:
        return HealthCheckResult(healthy=True)

    def _criteria(self, reading: _Reading) -> dict[str, dict[str, object]]:
        span = self._rubric.scale_max - _LOWEST
        scored = {}
        for criterion in self._rubric.criteria:
            fraction, why = HEURISTICS[criterion.name](reading)
            scored[criterion.name] = {"score": _LOWEST + round(fraction * span), "rationale": why}
        return scored


def _claims(reading: _Reading) -> list[dict[str, str]]:
    return [
        {
            "claim": line,
            "verdict": "supported" if reading.grounded(line) else "unsupported",
            "evidence": "",
        }
        for line in reading.cited
    ]
