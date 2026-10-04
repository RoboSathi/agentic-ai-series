"""The person's own message, screened before anything reads it: PII, injection, moderation.

The agent's other defences look at what the *web* says (`injection_classifier` over fetched
pages, the quarantined extractor). This one looks at what the *person* typed, through the
platform's `MessageScreener` — the same screens, and the same cards, as the Customer Support
agent's front door:

    PII filter          personal data is redacted; the redacted text is what flows on
    prompt injection    the phrase scan, then the hosted jailbreak model; a hit blocks the turn
    content moderation  the term scan, then the hosted safety model; a flag blocks the turn

What a block means is the platform's too: the message is quarantined for a person, a card says
so, and the person gets the platform's refusal. This agent only adds its audit entry.
"""

from pydantic import BaseModel

from agent_platform.logging import get_logger, instrument
from agent_platform.security import MessageScreener, ScreenSubject
from research_agent.observability.telemetry import RunTelemetry


class ScreenResult(BaseModel):
    """The message as it may flow on (redacted), and whether a screen stopped the turn."""

    text: str
    blocked: bool = False
    reason: str = ""
    # What the person is told when a screen stopped the turn (the platform's refusal).
    refusal: str = ""


@instrument("research_agent")
class InputScreen:
    """Runs the declared screens over one message, cheapest first; the first block stops it."""

    def __init__(self, screener: MessageScreener, telemetry: RunTelemetry) -> None:
        self._screener = screener
        self._telemetry = telemetry
        self._logger = get_logger("research_agent")

    async def screen(self, message: str, session_id: str = "") -> ScreenResult:
        reporter = self._telemetry.reporter
        subject = ScreenSubject(session_id=session_id, trace_id=self._telemetry.trace_id)
        text = self._screener.redact(message, reporter).text
        outcome = await self._screener.injection(text, reporter, subject=subject)
        if not outcome.blocked:
            outcome = await self._screener.moderation(text, reporter, subject=subject)
        if outcome.blocked:
            self._logger.warning("Input blocked", reason=outcome.reason)
            self._telemetry.audit("input.blocked", "deny", outcome.reason)
            return ScreenResult(
                text=text, blocked=True, reason=outcome.reason, refusal=outcome.refusal
            )
        return ScreenResult(text=text)


__all__ = ["InputScreen", "ScreenResult"]
