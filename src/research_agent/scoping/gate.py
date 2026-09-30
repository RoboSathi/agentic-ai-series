"""The scopability gate: can this first message become a bounded research question?

Three answers. `scopable` names an archetype and the interview starts on
its checklist. `needs_reframe` (an open-ended "tell me about X") makes the
first interview turn a question about purpose — the agent never proceeds
to research without one. `refuse` ends the conversation.

The classifier is a Protocol so the v1 implementation — a structured LLM
call — can be replaced by a small fine-tuned model without touching the
graph. Two deterministic filters run first: configured refusal patterns
(no model is asked whether to help build a weapon) and configured
open-ended patterns.
"""

import re
from typing import Protocol, runtime_checkable

from agentic_commons.logging import get_logger
from research_agent.config.settings import ScopingSettings
from research_agent.domain.scope import ARCHETYPES, GateVerdict
from research_agent.prompts import gate as gate_prompt
from research_agent.structured import StructuredCaller


@runtime_checkable
class ArchetypeClassifier(Protocol):
    """Classifies a first message into a gate verdict."""

    async def classify(self, message: str) -> GateVerdict: ...


class LlmArchetypeClassifier:
    """Deterministic refusal/reframe filters, then a structured model call."""

    def __init__(self, caller: StructuredCaller, settings: ScopingSettings) -> None:
        self._caller = caller
        self._cues = settings.archetype_keywords
        self._refuse = [re.compile(pattern, re.IGNORECASE) for pattern in settings.refuse_patterns]
        self._reframe = [
            re.compile(pattern, re.IGNORECASE) for pattern in settings.reframe_patterns
        ]
        self._logger = get_logger("research_agent")

    async def classify(self, message: str) -> GateVerdict:
        if any(pattern.search(message) for pattern in self._refuse):
            return GateVerdict(decision="refuse", reason="matches a refusal pattern")
        cues: dict[str, list[str]] = {name: self._cues.get(name, []) for name in ARCHETYPES}
        answer = await self._caller.call(gate_prompt.build(message, cues), gate_prompt.GateAnswer)
        verdict = GateVerdict.model_validate(answer.model_dump())
        if verdict.decision == "scopable" and any(p.search(message) for p in self._reframe):
            verdict = verdict.model_copy(update={"decision": "needs_reframe"})
        if verdict.decision == "scopable" and verdict.archetype is None:
            verdict = verdict.model_copy(update={"decision": "needs_reframe"})
        self._logger.info("Gate decision", decision=verdict.decision, archetype=verdict.archetype)
        return verdict
