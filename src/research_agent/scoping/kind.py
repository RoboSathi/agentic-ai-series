"""The research kind: which kind of research a `/research` question is, or a refusal.

`/research` has already settled that the user wants a research run; what is left is which kind,
because the kind picks the interview checklist, the rubric template, the stop rule and the
report's sections. `general` when no specialised kind fits.

The classifier is a Protocol so the LLM implementation can sit behind faster layers (cue rules,
the encoder: `kind_encoder.py`) without touching the graph. Deterministic rules wrap the model:
refusal patterns run first (no model is asked whether to help build a weapon), and the time window
is read from the message by the same parser the Brief uses.
"""

import re
from typing import Protocol, runtime_checkable

from agent_platform.logging import get_logger, instrument
from research_agent.config.settings import ScopingSettings
from research_agent.domain.scope import GENERAL, RESEARCH_ARCHETYPES, KindVerdict
from research_agent.llm.structured import StructuredCaller
from research_agent.prompts import kind as kind_prompt
from research_agent.scoping.window import TimeWindowParser


@runtime_checkable
class KindClassifier(Protocol):
    """Reads a `/research` question as a research kind (or refuses it)."""

    async def classify(self, message: str) -> KindVerdict: ...


@instrument("research_agent")
class LlmKindClassifier:
    """Deterministic refusal filter, a structured model call, deterministic clean-up."""

    def __init__(self, caller: StructuredCaller, settings: ScopingSettings) -> None:
        self._caller = caller
        self._settings = settings
        self._refuse = [re.compile(pattern, re.IGNORECASE) for pattern in settings.refuse_patterns]
        self._window = TimeWindowParser(settings.time_window)
        self._logger = get_logger("research_agent")

    def refused(self, message: str) -> bool:
        """Whether a refusal pattern matches — checked before any model is asked."""
        return any(pattern.search(message) for pattern in self._refuse)

    def window_days(self, message: str) -> int | None:
        """The time window the message states, read by rule (never by a model)."""
        return self._window.days(message)

    def normalise(self, verdict: KindVerdict, message: str) -> KindVerdict:
        """The window read by rule, and a kind that exists (`general` otherwise)."""
        days = self._window.days(message) or verdict.recent_days
        verdict = verdict.model_copy(update={"recent_days": days})
        if verdict.decision != "scopable":
            return verdict
        kind = verdict.archetype if verdict.archetype in RESEARCH_ARCHETYPES else GENERAL
        return verdict.model_copy(update={"archetype": kind, "topic": verdict.topic or message})

    async def classify(self, message: str) -> KindVerdict:
        if self.refused(message):
            return KindVerdict(decision="refuse", reason="matches a refusal pattern", source="rule")
        cues: dict[str, list[str]] = {
            name: self._settings.archetype_keywords.get(name, []) for name in RESEARCH_ARCHETYPES
        }
        prompt = kind_prompt.build(message, cues)
        answer = await self._caller.call(prompt, kind_prompt.KindAnswer)
        decided = KindVerdict.model_validate(answer.model_dump()).model_copy(
            update={"source": "llm", "source_model": self._caller.model_key(prompt.role)}
        )
        verdict = self.normalise(decided, message)
        self._logger.info(
            "Research kind decided",
            decision=verdict.decision,
            kind=verdict.archetype,
            recent_days=verdict.recent_days,
        )
        return verdict
