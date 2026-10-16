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

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from agent_platform.config import get_settings
from agent_platform.config.settings import ScreeningMessages
from agent_platform.logging import get_logger, instrument
from agent_platform.observability import checks
from agent_platform.observability.status_feed import StageReporter
from agent_platform.observability.types import EventKind
from agent_platform.resilience.dead_letter_queue.queue import DeadLetterQueue, DeadLetterReason
from agent_platform.security.content_moderation.types import ModerationVerdict
from agent_platform.security.pii.active import admit_if_on
from agent_platform.security.pii.detector import RedactedSpan
from agent_platform.security.pii.source import Admitted
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
        self._app_config = get_settings()
        self._suite = suite
        self._quarantine = quarantine
        self._logger = get_logger("security")

    def declared_checks(self) -> list[str]:
        """The checks this screener runs on every message: what `screen_coverage` expects.

        A composition root declares these for its agent (`units.declare_screens`).
        """
        guards = self._app_config.security.guards
        hosted = guards.enabled
        declared = {
            checks.PII: self._suite.pii is not None,
            checks.INJECTION: self._suite.injection is not None,
            checks.JAILBREAK: self._suite.injection is not None
            and hosted
            and guards.jailbreak.enabled
            and self._suite.jailbreak_guard is not None,
            checks.MODERATION: self._suite.moderator is not None,
            checks.CONTENT_SAFETY: self._suite.moderator is not None
            and hosted
            and guards.content_safety.enabled
            and self._suite.content_safety_guard is not None,
        }
        return [check for check, built in declared.items() if built]

    @property
    def suite(self) -> SecuritySuite:
        return self._suite

    def redact(self, text: str, reporter: StageReporter | None) -> PiiOutcome:
        """Personal data out of the message, as it enters, with its card. Never blocks.

        For a caller that screens the person's own words here. One that admitted them earlier
        (before its graph, so not even a checkpoint holds a value) calls `report` instead.
        """
        if self._suite.pii is None:
            return PiiOutcome(text=text)
        _started(reporter, PII_STAGE, _messages().pii_started)
        return self.report(admit_if_on(text), reporter)

    def report(self, admitted: Admitted, reporter: StageReporter | None) -> PiiOutcome:
        """The PII card for a message already admitted: what was tokenised — never a value."""
        if self._suite.pii is None:
            return PiiOutcome(text=admitted.text)
        words = _messages()
        spans = admitted.spans
        categories = admitted.categories
        summary = (
            words.pii_redacted.format(categories=", ".join(categories), count=len(spans))
            if spans
            else words.pii_clear
        )
        entities = [
            words.pii_entity.format(
                category=span.category,
                detector=span.detector,
                score=f"{span.score:.2f}",
                token=span.token,
            )
            for span in spans
        ]
        model, unavailable = self._suite.pii_model()
        checks.record_pii(model, unavailable, len(spans), admitted.latency_ms)
        fields = {
            **self._pii_checks(spans, admitted.latency_ms, words),
            "matches": len(spans),
            "categories": categories,
            "entities": entities,
        }
        self._logger.info("PII screen", **fields)
        _emit(reporter, PII_STAGE, "completed", summary, fields)
        return PiiOutcome(text=admitted.text, categories=categories, matches=len(spans))

    def _pii_checks(
        self, spans: Sequence[RedactedSpan], latency_ms: float, words: ScreeningMessages
    ) -> dict[str, Any]:
        """The same shape as the other screens: the rule-based recognizers, the model, the why.

        A span from a recognizer named in `security.pii.model_recognizers` (spaCy NER) is the
        model's; every other recognizer is a deterministic rule (patterns, checksums).
        """
        model_names = set(self._app_config.security.pii.model_recognizers)
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
        _started(reporter, INJECTION_STAGE, _messages().injection_started)
        assessed = await self._assess_injection(text, instructions, tools)
        return self._settled("prompt_injection", text, assessed.report(reporter), reporter, subject)

    async def moderation(
        self, text: str, reporter: StageReporter | None, subject: ScreenSubject | None = None
    ) -> ScreenOutcome:
        """The term scan, then the hosted safety model; a flag blocks and is quarantined."""
        if self._suite.moderator is None:
            return ScreenOutcome()
        _started(reporter, MODERATION_STAGE, _messages().moderation_started)
        assessed = await self._assess_moderation(text)
        return self._settled(
            "content_moderation", text, assessed.report(reporter), reporter, subject
        )

    async def screen_both(
        self,
        text: str,
        reporter: StageReporter | None,
        instructions: str = "",
        tools: Sequence[ToolDefinition] = (),
        subject: ScreenSubject | None = None,
    ) -> "ScreenPair":
        """Injection and moderation at once: one hosted round trip a message, not two.

        Both models are asked together; their verdicts are then taken in the order the two
        separate screens would have taken them. Injection decides first: when it blocks,
        moderation's verdict cannot change the outcome, so it is neither shown nor
        quarantined, and the cards read exactly as two screens in sequence would.
        """
        if self._suite.injection is None or self._suite.moderator is None:
            first = await self.injection(text, reporter, instructions, tools, subject)
            if first.blocked:
                return ScreenPair(injection=first)
            return ScreenPair(
                injection=first, moderation=await self.moderation(text, reporter, subject)
            )
        words = _messages()
        _started(reporter, INJECTION_STAGE, words.injection_started)
        injected, moderated = await asyncio.gather(
            self._assess_injection(text, instructions, tools), self._assess_moderation(text)
        )
        first = self._settled(
            "prompt_injection", text, injected.report(reporter), reporter, subject
        )
        if first.blocked:
            return ScreenPair(injection=first)
        _started(reporter, MODERATION_STAGE, words.moderation_started)
        second = self._settled(
            "content_moderation", text, moderated.report(reporter), reporter, subject
        )
        return ScreenPair(injection=first, moderation=second)

    async def _assess_injection(
        self, text: str, instructions: str, tools: Sequence[ToolDefinition]
    ) -> "_Assessed":
        started = time.perf_counter()
        assessment = await self._suite.prepare_untrusted_async(instructions, text, list(tools))
        rule = self._app_config.security.guards.rule_checks.injection
        return _Assessed(
            stage=INJECTION_STAGE,
            card=_injection_card(assessment, _messages()),
            outcome=ScreenOutcome(
                blocked=assessment.suspicious,
                reason=assessment.reason,
                allowed_actions=list(assessment.allowed_actions),
                checks=list(assessment.checks),
            ),
            record_rule=checks.INJECTION,
            rule_hit=any(
                entry.get("name") == rule and entry.get("status") == FAIL
                for entry in assessment.checks
            ),
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

    async def _assess_moderation(self, text: str) -> "_Assessed":
        started = time.perf_counter()
        verdict = await self._suite.moderate_async(text)
        return _Assessed(
            stage=MODERATION_STAGE,
            card=_moderation_card(verdict, _messages()),
            outcome=ScreenOutcome(
                blocked=verdict.flagged,
                reason=verdict.reason,
                categories=list(verdict.categories),
                checks=list(verdict.checks),
            ),
            record_rule=checks.MODERATION,
            rule_hit=bool(verdict.matched_terms),
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

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


class ScreenPair(BaseModel):
    """Both hosted screens' outcomes; `moderation` is None when injection already stopped it."""

    injection: ScreenOutcome
    moderation: ScreenOutcome | None = None

    @property
    def stopping(self) -> ScreenOutcome | None:
        """The screen that blocked the message, if one did."""
        for outcome in (self.injection, self.moderation):
            if outcome is not None and outcome.blocked:
                return outcome
        return None


@dataclass(frozen=True)
class _Assessed:
    """One screen's verdict, not yet reported: what it found, and the card that says so."""

    stage: str
    card: Card
    outcome: ScreenOutcome
    record_rule: str
    rule_hit: bool
    elapsed_ms: float

    def report(self, reporter: StageReporter | None) -> ScreenOutcome:
        """Counted for analytics and carded, then handed on to be settled."""
        checks.record_screen(
            self.outcome.checks,
            rule=self.record_rule,
            rule_hit=self.rule_hit,
            elapsed_ms=self.elapsed_ms,
        )
        status, summary, fields = self.card
        _emit(reporter, self.stage, status, summary, fields)
        return self.outcome


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
    asked. A flag shows what the model found (`extra`), or why when it named nothing — a
    `reason` that only restates the categories is noise."""
    if check is None:
        return "completed", passed_rule, fields
    shown = {"model": check.get("model"), "latency_ms": check.get("latency_ms")}
    status = check.get("status")
    found = {name: check.get(name) for name in extra if check.get(name) not in (None, "", [])}
    why = {} if found.get("categories") else {"reason": check.get("reason")}
    if status == FAIL:
        return "blocked", blocked, {**fields, **shown, **found, **why}
    if status == ADVISORY:
        return "warning", advisory, {**fields, **shown, **found, **why}
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
    "ScreenPair",
    "ScreenSubject",
    "MessageScreener",
    "PiiOutcome",
    "ScreenOutcome",
]
