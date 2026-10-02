"""The gate: a quick cited answer (Q&A), a scoped research run, or neither.

Four answers. `scopable` in mode `qa` goes straight to a small, automatic
run with no interview. `scopable` in mode `research` names the kind of
research (a specialised archetype, or `general`) and the interview starts
on its checklist. `needs_reframe` — a message with no question in it at
all — gets a numbered menu of the two modes. `refuse` ends it.

The classifier is a Protocol so the v1 implementation — a structured LLM
call — can be replaced by a small fine-tuned model without touching the
graph. Deterministic rules wrap the model: refusal patterns run first (no
model is asked whether to help build a weapon), and the time window is
read from the message by the same parser the Brief uses.
"""

import re
from typing import Protocol, runtime_checkable

from agentic_commons.logging import get_logger, instrument
from research_agent.config.settings import ReframeOption, ScopingSettings
from research_agent.domain.scope import (
    GENERAL,
    QA,
    RESEARCH_ARCHETYPES,
    GateVerdict,
    Mode,
)
from research_agent.prompts import gate as gate_prompt
from research_agent.scoping.window import TimeWindowParser
from research_agent.structured import StructuredCaller

_WORD_EDGE = r"(?<![\w&]){}(?![\w&])"


@runtime_checkable
class ArchetypeClassifier(Protocol):
    """Classifies a first message into a gate verdict.

    `forced` is the mode the user asked for (`/qa`, `/research`); without one the message is
    a plain question and is answered, never routed to research.
    """

    async def classify(self, message: str, forced: Mode | None = None) -> GateVerdict: ...


@instrument("research_agent")
class LlmArchetypeClassifier:
    """Deterministic refusal filter, a structured model call, deterministic clean-up."""

    def __init__(self, caller: StructuredCaller, settings: ScopingSettings) -> None:
        self._caller = caller
        self._settings = settings
        self._refuse = [re.compile(pattern, re.IGNORECASE) for pattern in settings.refuse_patterns]
        self._reframe = [
            re.compile(pattern, re.IGNORECASE) for pattern in settings.reframe_patterns
        ]
        self._window = TimeWindowParser(settings.time_window)
        self._logger = get_logger("research_agent")

    def refused(self, message: str) -> bool:
        """Whether a refusal pattern matches — checked before any model is asked."""
        return any(pattern.search(message) for pattern in self._refuse)

    def window_days(self, message: str) -> int | None:
        """The time window the message states, read by rule (never by a model)."""
        return self._window.days(message)

    def normalise(self, verdict: GateVerdict, message: str) -> GateVerdict:
        """The window read by rule, a reframe for a purposeless message, a kind that fits."""
        return self._normalise(verdict, message)

    def quick(self, message: str) -> GateVerdict:
        """A plain question: Q&A, with the time window it states. Decided in code, no model."""
        return GateVerdict(
            decision="scopable",
            mode="qa",
            archetype=QA,
            topic=message.strip(),
            reason=self._settings.quick_reason,
            source="rule",
            recent_days=self.window_days(message),
        )

    async def classify(self, message: str, forced: Mode | None = None) -> GateVerdict:
        if self.refused(message):
            return GateVerdict(decision="refuse", reason="matches a refusal pattern", source="rule")
        if forced != "research":
            return self.quick(message)
        cues: dict[str, list[str]] = {
            name: self._settings.archetype_keywords.get(name, []) for name in RESEARCH_ARCHETYPES
        }
        prompt = gate_prompt.build(message, cues, self._settings.research_cues)
        answer = await self._caller.call(prompt, gate_prompt.GateAnswer)
        decided = GateVerdict.model_validate(answer.model_dump()).model_copy(
            update={"source": "llm", "source_model": self._caller.model_key(prompt.role)}
        )
        verdict = self._normalise(decided, message)
        self._logger.info(
            "Gate decision",
            decision=verdict.decision,
            mode=verdict.mode,
            archetype=verdict.archetype,
            recent_days=verdict.recent_days,
        )
        return verdict

    def _normalise(self, verdict: GateVerdict, message: str) -> GateVerdict:
        """The window is read by rule; the kind always matches the mode."""
        days = self._window.days(message) or verdict.recent_days
        verdict = verdict.model_copy(update={"recent_days": days})
        if verdict.decision != "scopable":
            return verdict
        if verdict.mode == "qa" and days is None and self._purposeless(message):
            return verdict.model_copy(update={"decision": "needs_reframe"})
        return with_mode(verdict, verdict.mode)

    def _purposeless(self, message: str) -> bool:
        return any(pattern.search(message) for pattern in self._reframe)


def with_mode(verdict: GateVerdict, mode: Mode) -> GateVerdict:
    """A scopable verdict in `mode`, with a kind that fits it."""
    if mode == "qa":
        archetype = QA
    elif verdict.archetype in RESEARCH_ARCHETYPES:
        archetype = verdict.archetype
    else:
        archetype = GENERAL
    return verdict.model_copy(update={"decision": "scopable", "mode": mode, "archetype": archetype})


def render_options(options: list[ReframeOption], line: str) -> str:
    """The reframe menu, one numbered line per mode."""
    return "\n".join(
        line.format(number=number, label=option.label, example=option.example)
        for number, option in enumerate(options, start=1)
    )


def resolve_reframe_choice(reply: str, options: list[ReframeOption]) -> Mode | None:
    """A reply to the menu: its number, its label or one of its aliases — else None."""
    text = reply.strip().casefold()
    if text.isdigit() and 1 <= int(text) <= len(options):
        return options[int(text) - 1].mode
    for option in options:
        words = [option.label.casefold(), *(alias.casefold() for alias in option.aliases)]
        if any(re.search(_WORD_EDGE.format(re.escape(word)), text) for word in words):
            return option.mode
    return None
