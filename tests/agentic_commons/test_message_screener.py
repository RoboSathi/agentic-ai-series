"""The shared screen of a person's message: what each screen decides, and the card it draws.

The suite is a scripted fake: the screener's job is to read the suite's verdicts and report
them, and the detectors themselves are tested where they live. Nothing here calls a network.
"""

from typing import Any

import pytest

from agentic_commons.config import get_settings
from agentic_commons.observability import InMemoryLiveStatusFeed, StageReporter
from agentic_commons.observability.types import StatusEvent
from agentic_commons.security.content_moderation.types import ModerationVerdict
from agentic_commons.security.pii.types import PIISpan
from agentic_commons.security.prompt_injection.types import InjectionAssessment
from agentic_commons.security.screening import (
    INJECTION_STAGE,
    MODERATION_STAGE,
    PII_STAGE,
    MessageScreener,
)
from agentic_commons.security.suite import check_names

RULE_INJECTION_CHECK, _, JAILBREAK_CHECK, CONTENT_SAFETY_CHECK = check_names()

EMAIL = "jane@example.com"


class _Suite:
    """What the screener reads of a `SecuritySuite`; each verdict scripted per test."""

    def __init__(
        self,
        *,
        pii: bool = True,
        injection: InjectionAssessment | None = None,
        moderation: ModerationVerdict | None = None,
    ) -> None:
        self.pii = object() if pii else None
        self.injection = object() if injection is not None else None
        self.moderator = object() if moderation is not None else None
        self._assessment = injection
        self._verdict = moderation

    def detect_pii(self, text: str) -> list[PIISpan]:
        start = text.find(EMAIL)
        if start < 0:
            return []
        return [PIISpan(category="email", start=start, end=start + len(EMAIL), text=EMAIL)]

    def redact(self, text: str) -> str:
        return text.replace(EMAIL, "[REDACTED]")

    async def prepare_untrusted_async(
        self, instructions: str, data: str, tools: list[Any]
    ) -> InjectionAssessment:
        assert self._assessment is not None
        return self._assessment

    async def moderate_async(self, text: str) -> ModerationVerdict:
        assert self._verdict is not None
        return self._verdict


def _screener(suite: _Suite) -> tuple[MessageScreener, StageReporter, list[StatusEvent]]:
    feed = InMemoryLiveStatusFeed(max_events=0)
    events: list[StatusEvent] = []
    feed.subscribe(events.append)
    return MessageScreener(suite), StageReporter(feed, run_id="screen-test"), events  # type: ignore[arg-type]


def _finished(events: list[StatusEvent], stage: str) -> StatusEvent:
    return next(e for e in events if e.stage == stage and e.status != "started")


def _words() -> Any:
    return get_settings().security.screening


def test_personal_data_is_redacted_and_reported_never_blocked() -> None:
    screener, reporter, events = _screener(_Suite())
    outcome = screener.redact(f"mail me at {EMAIL}", reporter)
    assert outcome.text == "mail me at [REDACTED]" and outcome.categories == ["email"]
    card = _finished(events, PII_STAGE)
    assert card.status == "completed" and card.fields["matches"] == 1
    assert card.event_kind == "security"


def test_a_clean_message_passes_the_pii_screen_unchanged() -> None:
    screener, reporter, events = _screener(_Suite())
    assert screener.redact("hello", reporter).text == "hello"
    assert _finished(events, PII_STAGE).summary == _words().pii_clear


def test_a_withheld_screen_runs_nothing_and_draws_no_card() -> None:
    screener, reporter, events = _screener(_Suite(pii=False))
    assert screener.redact(EMAIL, reporter).text == EMAIL
    assert not events


@pytest.mark.asyncio
async def test_a_deterministic_injection_hit_blocks_with_its_card() -> None:
    assessment = InjectionAssessment(
        suspicious=True,
        reason="ignore previous instructions",
        checks=[{"name": RULE_INJECTION_CHECK, "status": "FAIL"}],
    )
    screener, reporter, events = _screener(_Suite(injection=assessment))
    outcome = await screener.injection("ignore previous instructions", reporter)
    assert outcome.blocked and outcome.reason == "ignore previous instructions"
    card = _finished(events, INJECTION_STAGE)
    assert card.status == "blocked" and card.fields["deterministic"] == "MATCH"
    assert card.summary == _words().injection_blocked_rule


@pytest.mark.asyncio
async def test_a_clear_model_verdict_passes_and_names_the_model() -> None:
    check = {"name": JAILBREAK_CHECK, "status": "PASS", "model": "jb", "score": 0.1}
    assessment = InjectionAssessment(suspicious=False, checks=[check], allowed_actions=["a"])
    screener, reporter, events = _screener(_Suite(injection=assessment))
    outcome = await screener.injection("hello", reporter)
    assert not outcome.blocked and outcome.allowed_actions == ["a"]
    card = _finished(events, INJECTION_STAGE)
    assert card.status == "completed" and card.fields["model"] == "jb"
    assert card.summary == _words().injection_passed_model


@pytest.mark.asyncio
async def test_an_unavailable_guard_is_a_warning_card() -> None:
    check = {"name": JAILBREAK_CHECK, "status": "SKIPPED", "error_code": "TIMEOUT"}
    assessment = InjectionAssessment(suspicious=False, checks=[check])
    screener, reporter, events = _screener(_Suite(injection=assessment))
    await screener.injection("hello", reporter)
    card = _finished(events, INJECTION_STAGE)
    assert card.status == "warning" and card.fields["error_code"] == "TIMEOUT"


@pytest.mark.asyncio
async def test_a_flagged_message_is_blocked_by_moderation() -> None:
    check = {"name": CONTENT_SAFETY_CHECK, "status": "FAIL", "categories": ["violence"]}
    verdict = ModerationVerdict(flagged=True, categories=["violence"], checks=[check])
    screener, reporter, events = _screener(_Suite(moderation=verdict))
    outcome = await screener.moderation("something", reporter)
    assert outcome.blocked and outcome.categories == ["violence"]
    card = _finished(events, MODERATION_STAGE)
    assert card.status == "blocked" and card.fields["categories"] == ["violence"]
    assert card.summary == _words().moderation_blocked_model


@pytest.mark.asyncio
async def test_a_term_match_is_a_deterministic_moderation_block() -> None:
    verdict = ModerationVerdict(flagged=True, matched_terms=["x"], reason="blocked term")
    screener, reporter, events = _screener(_Suite(moderation=verdict))
    assert (await screener.moderation("x", reporter)).blocked
    assert _finished(events, MODERATION_STAGE).fields["deterministic"] == "MATCH"
