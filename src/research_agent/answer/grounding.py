"""Checking a tool-backed answer against what its tools returned, before the user reads it.

Two checks, both on the answer's own sentences:

    the platform's response guardrail   claims against the evidence: a figure or date no tool
                                        returned, a claim the evidence contradicts; one bounded
                                        repair (the model rewrites without them), then fail closed
    unsourced names                     a multi-word proper name that no tool returned and the
                                        question did not contain ("Rohit Sharma", a stadium) is
                                        invented, and a model can only check that by looking

The first is `agentic_commons.guardrails.ResponseGuardrail`, wired as the support agent wires it;
the second is this agent's, because no claim ladder can tell that a name is missing.
"""

import re
from collections.abc import Awaitable, Callable

from pydantic import BaseModel, Field

from agentic_commons.context.types import Chunk
from agentic_commons.guardrails.response import ResponseGuardrail
from agentic_commons.logging import get_logger, instrument
from research_agent.config.settings import AnswerGroundingSettings, AnswerMessages

Revise = Callable[[str, list[str]], Awaitable[str]]
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n+")


class GroundingOutcome(BaseModel):
    """What the check did to an answer, for the reply and for the card."""

    text: str
    claims: int = 0
    supported: int = 0
    repaired: bool = False
    # Claims (and sentences) the sources do not support, and were taken out.
    removed: list[str] = Field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.removed


class _Assessment(BaseModel):
    """One pass over one text: what survives, what was refused, how many claims were read."""

    kept: str
    removed: list[str] = Field(default_factory=list)
    claims: int = 0
    supported: int = 0
    escalated: bool = False

    @property
    def clean(self) -> bool:
        return not self.removed and not self.escalated


@instrument("research_agent")
class AnswerGrounding:
    """The guardrail, then the unsourced-name check; one rewrite from the sources if either objects."""

    def __init__(
        self,
        guardrail: ResponseGuardrail,
        settings: AnswerGroundingSettings,
        messages: AnswerMessages,
    ) -> None:
        self._guardrail = guardrail
        self._names = re.compile(settings.entity_pattern)
        self._messages = messages
        self._logger = get_logger("research_agent")

    async def check(
        self, question: str, answer: str, evidence: list[Chunk], revise: Revise
    ) -> GroundingOutcome:
        """The answer as it may be shown: untouched, rewritten from the sources, or filtered."""
        first = await self._assess(question, answer, evidence)
        if first.clean:
            return self._outcome(answer, first)
        best, repaired = first, False
        rewritten = await self._rewritten(answer, first.removed, revise)
        if rewritten and rewritten != answer.strip():
            second = await self._assess(question, rewritten, evidence)
            if second.clean:
                return self._outcome(rewritten, second, removed=first.removed, repaired=True)
            if len(second.removed) < len(first.removed):
                best, repaired = second, True
        return self._outcome(self._filtered(best), best, repaired=repaired)

    async def _rewritten(self, answer: str, removed: list[str], revise: Revise) -> str:
        """The model's rewrite from the sources; empty when it cannot give one (the answer is
        then filtered instead: a model that stalls here must not cost the user the answer)."""
        try:
            return (await revise(answer, removed)).strip()
        except Exception as exc:
            self._logger.warning("Grounding rewrite unavailable; filtering instead", error=str(exc))
            return ""

    async def _assess(self, question: str, text: str, evidence: list[Chunk]) -> _Assessment:
        result = await self._guardrail.enforce(text, evidence)
        survivors = result.response if result.supported or not result.escalated else ""
        kept, invented = self._without_invented_names(survivors, question, evidence)
        return _Assessment(
            kept=kept,
            removed=[*result.unsupported_claims, *invented],
            claims=result.claims,
            supported=result.supported,
            escalated=result.escalated,
        )

    def _filtered(self, assessment: _Assessment) -> str:
        """What is left of a text once its unsupported sentences are gone, with a note saying so."""
        if not assessment.kept.strip():
            return self._messages.withheld
        return f"{assessment.kept}\n\n{self._messages.filtered_note}"

    @staticmethod
    def _outcome(
        text: str,
        assessment: _Assessment,
        removed: list[str] | None = None,
        repaired: bool = False,
    ) -> GroundingOutcome:
        return GroundingOutcome(
            text=text,
            claims=assessment.claims,
            supported=assessment.supported,
            repaired=repaired,
            removed=assessment.removed if removed is None else removed,
        )

    def _without_invented_names(
        self, text: str, question: str, evidence: list[Chunk]
    ) -> tuple[str, list[str]]:
        """The text without any sentence that names someone or something no tool returned."""
        known = "\n".join([question, *(chunk.content for chunk in evidence)]).casefold()
        kept: list[str] = []
        invented: list[str] = []
        for sentence in _SENTENCE_BREAK.split(text):
            if not sentence.strip():
                continue
            unknown = [n for n in self._names.findall(sentence) if n.casefold() not in known]
            (invented if unknown else kept).append(sentence.strip())
        return "\n".join(kept), invented
