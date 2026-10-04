"""The person's message, screened before any model sees it: PII, prompt injection, moderation.

One screener for every agent: the detectors are the `SecuritySuite`'s, and each screen that
runs draws the same card wherever it runs (`pii_filtering`, `prompt_injection_check`,
`content_moderation`). What a block means is the same everywhere too, so it is here: the
message (PII redacted) is quarantined in the dead-letter queue for a person, one
`dead_letter_queue` card says so, and the outcome carries the one configured refusal. An
agent only stops its turn and says `outcome.refusal`.

A screen whose detector was withheld (`suite.pii`, `suite.injection`, `suite.moderator` is
None) is never run and draws no card: the trace shows the gap instead of a check that passed
without looking at anything.
"""

import time
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field

from agent_platform.config import get_settings
from agent_platform.config.settings import ScreeningMessages
from agent_platform.logging import get_logger, instrument
from agent_platform.observability.status_feed import StageReporter
from agent_platform.observability.types import EventKind
from agent_platform.resilience.dead_letter_queue.queue import DeadLetterQueue, DeadLetterReason
from agent_platform.security.content_moderation.types import ModerationVerdict
from agent_platform.security.pii.types import PIISpan
from agent_platform.security.prompt_injection.types import InjectionAssessment
from agent_platform.security.suite import SecuritySuite, check_names
from agent_platform.tools.types import ToolDefinition

PII_STAGE = "pii_filtering"
INJECTION_STAGE = "prompt_injection_check"
MODERATION_STAGE = "content_moderation"
QUARANTINE_STAGE = "dead_letter_queue"
SECURITY_KIND: EventKind = "security"
QUARANTINE_KIND: EventKind = "escalation"
# A deterministic scan's verdict, as the card shows it.
MATCH, NO_MATCH = "MATCH", "NO_MATCH"
# A hosted check's status, as the guards report it.
FAIL, SKIPPED, ADVISORY = "FAIL", "SKIPPED", "ADVISORY"


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
    # When blocked: what the person is told, and the quarantine entry a reviewer will find.
    refusal: str = ""
    quarantine_id: int | None = None
    # Every check that ran, with all it found (the model's reply included): what a reviewer of
    # the quarantine entry needs to judge the block.
    checks: list[dict[str, Any]] = Field(default_factory=list)


class ScreenSubject(BaseModel):
    """Whose message this is and where it came from, for the quarantine entry."""

    session_id: str = ""
    trace_id: str = ""
    # The agent's principal (a customer, a user), opaque here.
    principal_id: str | None = None


@instrument("security")
class MessageScreener:
    """PII, injection and moderation over one message, each reported as its own card."""

    def __init__(self, suite: SecuritySuite, quarantine: DeadLetterQueue | None = None) -> None:
        self._suite = suite
        self._quarantine = quarantine
        self._logger = get_logger("security")

    @property
    def suite(self) -> SecuritySuite:
        return self._suite

    def redact(self, text: str, reporter: StageReporter | None) -> PiiOutcome:
        """Personal data out of the message. Never blocks: a phone number is not an attack."""
        if self._suite.pii is None:
            return PiiOutcome(text=text)
        words = _messages()
        _started(reporter, PII_STAGE, words.pii_started)
        started = time.perf_counter()
        spans = self._suite.detect_pii(text)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        categories = sorted({span.category for span in spans})
        summary = (
            words.pii_redacted.format(categories=", ".join(categories), count=len(spans))
            if spans
            else words.pii_clear
        )
        # Each redaction as it may be shown: category, recognizer, score, token — never a value.
        entities = [
            words.pii_entity.format(
                category=span.category,
                detector=span.detector,
                score=f"{span.score:.2f}",
                token=span.token,
            )
            for span in self._suite.pii_tokens(text)
        ] or [
            words.pii_entity_masked.format(
                category=span.category, detector=span.detector, score=f"{span.score:.2f}"
            )
            for span in spans
        ]
        fields = {
            **self._pii_checks(spans, latency_ms, words),
            "matches": len(spans),
            "categories": categories,
            "entities": entities,
        }
        self._logger.info("PII screen", **fields)
        _emit(reporter, PII_STAGE, "completed", summary, fields)
        redacted = self._suite.redact(text) if spans else text
        return PiiOutcome(text=redacted, categories=categories, matches=len(spans))

    def _pii_checks(
        self, spans: list[PIISpan], latency_ms: float, words: ScreeningMessages
    ) -> dict[str, Any]:
        """The same shape as the other screens: the rule-based recognizers, the model, the why.

        A span from a recognizer named in `security.pii.model_recognizers` (spaCy NER) is the
        model's; every other recognizer is a deterministic rule (patterns, checksums).
        """
        model_names = set(get_settings().security.pii.model_recognizers)
        by_model = [span for span in spans if span.detector in model_names]
        by_rule = [span for span in spans if span.detector not in model_names]
        model, unavailable = self._suite.pii_model()
        reason = (
            words.pii_reason_unavailable.format(error=unavailable)
            if unavailable
            else words.pii_reason_found.format(
                rule=len(by_rule),
                model=len(by_model),
                categories=", ".join(sorted({span.category for span in spans})),
            )
            if spans
            else words.pii_reason_clear
        )
        return {
            "deterministic": MATCH if by_rule else NO_MATCH,
            "model": model or words.pii_no_model,
            "model_result": MATCH if by_model else NO_MATCH,
            "latency_ms": latency_ms,
            "reason": reason,
        }

    async def injection(
        self,
        text: str,
        reporter: StageReporter | None,
        instructions: str = "",
        tools: Sequence[ToolDefinition] = (),
        subject: ScreenSubject | None = None,
    ) -> ScreenOutcome:
        """The phrase scan, then the hosted jailbreak model; a hit blocks and is quarantined."""
        if self._suite.injection is None:
            return ScreenOutcome(allowed_actions=[tool.name for tool in tools])
        words = _messages()
        _started(reporter, INJECTION_STAGE, words.injection_started)
        assessment = await self._suite.prepare_untrusted_async(instructions, text, list(tools))
        status, summary, fields = _injection_card(assessment, words)
        _emit(reporter, INJECTION_STAGE, status, summary, fields)
        outcome = ScreenOutcome(
            blocked=assessment.suspicious,
            reason=assessment.reason,
            allowed_actions=list(assessment.allowed_actions),
            checks=list(assessment.checks),
        )
        return self._settled("prompt_injection", text, outcome, reporter, subject)

    async def moderation(
        self, text: str, reporter: StageReporter | None, subject: ScreenSubject | None = None
    ) -> ScreenOutcome:
        """The term scan, then the hosted safety model; a flag blocks and is quarantined."""
        if self._suite.moderator is None:
            return ScreenOutcome()
        words = _messages()
        _started(reporter, MODERATION_STAGE, words.moderation_started)
        verdict = await self._suite.moderate_async(text)
        status, summary, fields = _moderation_card(verdict, words)
        _emit(reporter, MODERATION_STAGE, status, summary, fields)
        outcome = ScreenOutcome(
            blocked=verdict.flagged,
            reason=verdict.reason,
            categories=list(verdict.categories),
            checks=list(verdict.checks),
        )
        return self._settled("content_moderation", text, outcome, reporter, subject)

    def _settled(
        self,
        reason: DeadLetterReason,
        text: str,
        outcome: ScreenOutcome,
        reporter: StageReporter | None,
        subject: ScreenSubject | None,
    ) -> ScreenOutcome:
        """A passed screen as it is; a blocked one quarantined, carded, and given its refusal."""
        if not outcome.blocked:
            return outcome
        words = _messages()
        entry_id, error = self._quarantined(reason, text, outcome, subject or ScreenSubject())
        summary = (
            words.quarantine_failed.format(error=error, reason=outcome.reason)
            if error
            else words.quarantined.format(reason=outcome.reason)
        )
        fields = {
            "screen": reason,
            "reason": outcome.reason,
            "categories": outcome.categories,
            "queued": entry_id is not None,
            "queue_id": entry_id,
        }
        if reporter is not None:
            reporter.emit(
                QUARANTINE_STAGE,
                "blocked",
                event_kind=QUARANTINE_KIND,
                summary=summary,
                fields=fields,
            )
        return outcome.model_copy(update={"refusal": words.refusal, "quarantine_id": entry_id})

    def _entry_metadata(
        self, screen: DeadLetterReason, text: str, outcome: ScreenOutcome
    ) -> dict[str, Any]:
        """What a reviewer needs: which screen, what decided, every check, and the PII tokens
        that let an authorised reviewer recover the redacted words (`make pii-reveal`)."""
        failed = [check for check in outcome.checks if check.get("status") == FAIL]
        decided_by = [str(check.get("name", "")) for check in failed]
        return {
            "screen": screen,
            "reason": outcome.reason,
            "categories": outcome.categories,
            "decided_by": decided_by,
            "checks": outcome.checks,
            "pii_tokens": [span._asdict() for span in self._suite.pii_tokens(text)],
        }

    def _quarantined(
        self, reason: DeadLetterReason, text: str, outcome: ScreenOutcome, subject: ScreenSubject
    ) -> tuple[int | None, str]:
        """The quarantine entry's id, or the store's error. A failed store still blocks."""
        if self._quarantine is None or not self._quarantine.enabled:
            return None, ""
        try:
            entry = self._quarantine.enqueue(
                reason=reason,
                payload=self._suite.redact(text),
                trace_id=subject.trace_id,
                session_id=subject.session_id,
                principal_id=subject.principal_id,
                metadata=self._entry_metadata(reason, text, outcome),
            )
        except Exception as exc:
            self._logger.error("Quarantine store failed; the message stays blocked", error=str(exc))
            return None, str(exc)
        return (entry.id if entry is not None else None), ""


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
        extra=("categories", "output"),
        advisory=words.moderation_advisory,
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
    advisory: str = "",
) -> Card:
    """The hosted model's part of a screen: blocked, advisory, unavailable, passed, or never
    asked. A flag shows what the model said (`extra`) and why (`reason`)."""
    if check is None:
        return "completed", passed_rule, fields
    shown = {"model": check.get("model"), "latency_ms": check.get("latency_ms")}
    status = check.get("status")
    found = {name: check.get(name) for name in extra if check.get(name) not in (None, "")}
    if status == FAIL:
        return "blocked", blocked, {**fields, **shown, **found, "reason": check.get("reason")}
    if status == ADVISORY:
        return "warning", advisory, {**fields, **shown, **found, "reason": check.get("reason")}
    if status == SKIPPED:
        failure = {name: check.get(name) for name in ("error_code", "http_status", "reason")}
        return "warning", unavailable, {**fields, **shown, **failure}
    passed = {name: value for name, value in found.items() if name != "categories"}
    return "completed", passed_model, {**fields, **shown, **passed}


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
    "QUARANTINE_STAGE",
    "ScreenSubject",
    "MessageScreener",
    "PiiOutcome",
    "ScreenOutcome",
]
