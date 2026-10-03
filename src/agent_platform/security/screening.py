"""The person's message, screened before any model sees it: PII, prompt injection, moderation.

One screener for every agent: the detectors are the `SecuritySuite`'s, and each screen that
runs draws the same card wherever it runs (`pii_filtering`, `prompt_injection_check`,
`content_moderation`). The screener only reports and decides *blocked or not*; what an agent
does with a block — quarantine it, refuse it, audit it — is the agent's own business.

A screen whose detector was withheld (`suite.pii`, `suite.injection`, `suite.moderator` is
None) is never run and draws no card: the trace shows the gap instead of a check that passed
without looking at anything.
"""

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field

from agent_platform.config import get_settings
from agent_platform.config.settings import ScreeningMessages
from agent_platform.logging import instrument
from agent_platform.observability.status_feed import StageReporter
from agent_platform.observability.types import EventKind
from agent_platform.security.content_moderation.types import ModerationVerdict
from agent_platform.security.prompt_injection.types import InjectionAssessment
from agent_platform.security.suite import SecuritySuite, check_names
from agent_platform.tools.types import ToolDefinition

PII_STAGE = "pii_filtering"
INJECTION_STAGE = "prompt_injection_check"
MODERATION_STAGE = "content_moderation"
SECURITY_KIND: EventKind = "security"
# A deterministic scan's verdict, as the card shows it.
MATCH, NO_MATCH = "MATCH", "NO_MATCH"
# A hosted check's status, as the guards report it.
FAIL, SKIPPED = "FAIL", "SKIPPED"


class PiiOutcome(BaseModel):
    """The message as it may leave the process (redacted), and what was taken out."""

    text: str
    categories: list[str] = Field(default_factory=list)
    matches: int = 0


class ScreenOutcome(BaseModel):
    """Whether a screen stopped the message, and why."""

    blocked: bool = False
    reason: str = ""
    categories: list[str] = Field(default_factory=list)
    # The tools still callable after an injection assessment (all of them when clear).
    allowed_actions: list[str] = Field(default_factory=list)


@instrument("security")
class MessageScreener:
    """PII, injection and moderation over one message, each reported as its own card."""

    def __init__(self, suite: SecuritySuite) -> None:
        self._suite = suite

    @property
    def suite(self) -> SecuritySuite:
        return self._suite

    def redact(self, text: str, reporter: StageReporter | None) -> PiiOutcome:
        """Personal data out of the message. Never blocks: a phone number is not an attack."""
        if self._suite.pii is None:
            return PiiOutcome(text=text)
        words = _messages()
        _started(reporter, PII_STAGE, words.pii_started)
        spans = self._suite.detect_pii(text)
        categories = sorted({span.category for span in spans})
        summary = (
            words.pii_redacted.format(categories=", ".join(categories), count=len(spans))
            if spans
            else words.pii_clear
        )
        _emit(
            reporter,
            PII_STAGE,
            "completed",
            summary,
            {"matches": len(spans), "categories": categories},
        )
        redacted = self._suite.redact(text) if spans else text
        return PiiOutcome(text=redacted, categories=categories, matches=len(spans))

    async def injection(
        self,
        text: str,
        reporter: StageReporter | None,
        instructions: str = "",
        tools: Sequence[ToolDefinition] = (),
    ) -> ScreenOutcome:
        """The phrase scan, then the hosted jailbreak model; a hit blocks."""
        if self._suite.injection is None:
            return ScreenOutcome(allowed_actions=[tool.name for tool in tools])
        words = _messages()
        _started(reporter, INJECTION_STAGE, words.injection_started)
        assessment = await self._suite.prepare_untrusted_async(instructions, text, list(tools))
        status, summary, fields = _injection_card(assessment, words)
        _emit(reporter, INJECTION_STAGE, status, summary, fields)
        return ScreenOutcome(
            blocked=assessment.suspicious,
            reason=assessment.reason,
            allowed_actions=list(assessment.allowed_actions),
        )

    async def moderation(self, text: str, reporter: StageReporter | None) -> ScreenOutcome:
        """The term scan, then the hosted safety model; a flag blocks."""
        if self._suite.moderator is None:
            return ScreenOutcome()
        words = _messages()
        _started(reporter, MODERATION_STAGE, words.moderation_started)
        verdict = await self._suite.moderate_async(text)
        status, summary, fields = _moderation_card(verdict, words)
        _emit(reporter, MODERATION_STAGE, status, summary, fields)
        return ScreenOutcome(
            blocked=verdict.flagged, reason=verdict.reason, categories=list(verdict.categories)
        )


# -- the cards ------------------------------------------------------------------------------

Card = tuple[str, str, dict[str, Any]]


def _injection_card(assessment: InjectionAssessment, words: ScreeningMessages) -> Card:
    rule_name, _, model_name, _ = check_names()
    model = _check(assessment.checks, model_name)
    rule = _check(assessment.checks, rule_name)
    deterministic = rule is not None and rule.get("status") == FAIL
    fields: dict[str, Any] = {"deterministic": MATCH if deterministic else NO_MATCH}
    if deterministic:
        return "blocked", words.injection_blocked_rule, {**fields, "reason": assessment.reason}
    return _model_card(
        model,
        fields,
        blocked=words.injection_blocked_model,
        unavailable=words.injection_unavailable,
        passed_model=words.injection_passed_model,
        passed_rule=words.injection_passed_rule,
        extra=("score",),
    )


def _moderation_card(verdict: ModerationVerdict, words: ScreeningMessages) -> Card:
    model = _check(verdict.checks, check_names()[3])
    fields: dict[str, Any] = {"deterministic": MATCH if verdict.matched_terms else NO_MATCH}
    if verdict.matched_terms:
        return "blocked", words.moderation_blocked_rule, {**fields, "reason": verdict.reason}
    return _model_card(
        model,
        fields,
        blocked=words.moderation_blocked_model,
        unavailable=words.moderation_unavailable,
        passed_model=words.moderation_passed_model,
        passed_rule=words.moderation_passed_rule,
        extra=("categories",),
    )


def _model_card(
    check: dict[str, Any] | None,
    fields: dict[str, Any],
    *,
    blocked: str,
    unavailable: str,
    passed_model: str,
    passed_rule: str,
    extra: tuple[str, ...],
) -> Card:
    """The hosted model's part of a screen: blocked, unavailable, passed, or never asked."""
    if check is None:
        return "completed", passed_rule, fields
    shown = {"model": check.get("model"), "latency_ms": check.get("latency_ms")}
    status = check.get("status")
    if status == FAIL:
        found = {name: check.get(name) for name in extra}
        return "blocked", blocked, {**fields, **shown, **found, "reason": check.get("reason")}
    if status == SKIPPED:
        failure = {name: check.get(name) for name in ("error_code", "http_status", "reason")}
        return "warning", unavailable, {**fields, **shown, **failure}
    found = {name: check.get(name) for name in extra if name != "categories"}
    return "completed", passed_model, {**fields, **shown, **found}


def _check(checks: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    return next((check for check in checks if check.get("name") == name), None)


# -- reporting ------------------------------------------------------------------------------


def _messages() -> ScreeningMessages:
    return get_settings().security.screening


def _started(reporter: StageReporter | None, stage: str, summary: str) -> None:
    if reporter is not None:
        reporter.started(stage, event_kind=SECURITY_KIND, summary=summary)


def _emit(
    reporter: StageReporter | None,
    stage: str,
    status: Any,
    summary: str,
    fields: dict[str, Any],
) -> None:
    if reporter is not None:
        reporter.emit(stage, status, event_kind=SECURITY_KIND, summary=summary, fields=fields)


__all__ = [
    "INJECTION_STAGE",
    "MODERATION_STAGE",
    "PII_STAGE",
    "MessageScreener",
    "PiiOutcome",
    "ScreenOutcome",
]
