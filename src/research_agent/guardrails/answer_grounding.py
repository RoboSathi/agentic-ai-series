"""Checking a tool-backed answer against what its tools returned, before the user reads it.

Two checks, on the answer's own sentences:

    the platform's response guardrail   each claim against the tool results it may be grounded
                                        on (`agentic_commons.guardrails.ResponseGuardrail`, wired
                                        as the support agent wires it): retrieval then NLI, a
                                        figure or date no tool returned, one bounded rewrite of
                                        what it refuted, then the refuted sentences removed
    unsourced names                     a multi-word proper name that no tool returned and the
                                        question did not contain ("Rohit Sharma", a stadium) is
                                        invented; its sentence is removed, with no model call

The answer's own "Sources:" block lists URLs, not claims: it is set aside before the check and
put back after it.
"""

import re
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, Field

from agentic_commons.config import get_settings
from agentic_commons.context.types import Chunk
from agentic_commons.guardrails.response import ResponseGuardrail, ResponseGuardrailResult
from agentic_commons.hallucination_protection import sentence_pieces
from agentic_commons.logging import instrument
from research_agent.config.settings import AnswerGroundingSettings, AnswerMessages

Revise = Callable[[str, list[str]], Awaitable[str]]
_WORD = re.compile(r"\w+")


class GroundingOutcome(BaseModel):
    """What the check did to an answer: the text to send, and the card's fields."""

    text: str
    # Sentences taken out: claims the guardrail refuted, then names no tool returned.
    removed: list[str] = Field(default_factory=list)
    repaired: bool = False
    withheld: bool = False
    # The guardrail's own report: every claim, its verdict, and the evidence behind it.
    report: dict[str, Any] = Field(default_factory=dict)

    @property
    def clean(self) -> bool:
        return not self.removed


@instrument("research_agent")
class AnswerGrounding:
    """The platform guardrail, then the unsourced-name check."""

    def __init__(
        self,
        guardrail: ResponseGuardrail,
        settings: AnswerGroundingSettings,
        messages: AnswerMessages,
    ) -> None:
        self._guardrail = guardrail
        self._names = re.compile(settings.entity_pattern)
        self._sources = re.compile(settings.sources_pattern)
        self._unknown_share = settings.name_unknown_share
        self._withheld_reply = get_settings().hallucination_protection.withheld_reply.strip()
        self._messages = messages

    async def check(
        self, question: str, answer: str, evidence: list[Chunk], revise: Revise
    ) -> GroundingOutcome:
        """The answer as it may be shown: untouched, rewritten from the sources, or filtered."""
        body, sources = self._split_sources(answer)
        result = await self._guardrail.enforce(body, evidence, revise=revise)
        # Nothing survived when the guardrail answers with its own withheld reply.
        verified = "" if result.response.strip() == self._withheld_reply else result.response
        kept, invented = self._without_invented_names(verified, question, evidence)
        removed = [*result.unsupported_claims, *invented]
        withheld = not kept.strip()
        text = self._shown(kept, sources, filtered=bool(removed), withheld=withheld)
        return GroundingOutcome(
            text=text,
            removed=removed,
            repaired=result.repaired,
            withheld=withheld,
            report=self._report(result, removed, withheld),
        )

    def _split_sources(self, answer: str) -> tuple[str, str]:
        """The answer's claims, and its Sources block (kept apart: URLs are not claims)."""
        found = self._sources.search(answer)
        if found is None:
            return answer.strip(), ""
        return answer[: found.start()].strip(), answer[found.start() :].strip()

    def _shown(self, kept: str, sources: str, *, filtered: bool, withheld: bool) -> str:
        if withheld:
            return self._messages.withheld
        parts = [kept.strip(), self._messages.filtered_note if filtered else "", sources]
        return "\n\n".join(part for part in parts if part)

    def _report(
        self, result: ResponseGuardrailResult, removed: list[str], withheld: bool
    ) -> dict[str, Any]:
        """The guardrail's fields for the card, with the outcome in this agent's words."""
        notes = self._messages
        outcome = (
            notes.outcome_withheld
            if withheld
            else notes.outcome_filtered
            if removed
            else notes.outcome_repaired
            if result.repaired
            else notes.outcome_unchanged
        )
        return {
            **result.model_dump(mode="json"),
            "outcome": outcome,
            "removed": removed,
            # Blocking is the support agent's word for a refused reply; here it means removed.
            "escalated": bool(removed) or withheld,
        }

    def _invented(self, name: str, known: set[str]) -> bool:
        """Whether most of a name's words appear in no source.

        Word by word, not as written: "Chief Global Affairs Officer Chris Lehane" is sourced by
        "Chris Lehane, Chief Global Affairs Officer". And by share, not any one word: "UN
        Security Council" against "United Nations Security Council", or a sentence that opens
        on a capitalised word ("Key AI news"), is not an invention; "Rohit Sharma" in a match
        report that names neither word is.
        """
        words = set(_WORD.findall(name.casefold()))
        return bool(words) and len(words - known) / len(words) > self._unknown_share

    def _without_invented_names(
        self, text: str, question: str, evidence: list[Chunk]
    ) -> tuple[str, list[str]]:
        """The text without any sentence that names someone or something no tool returned."""
        known = set(_WORD.findall("\n".join([question, *(c.content for c in evidence)]).casefold()))
        kept: list[str] = []
        invented: list[str] = []
        for sentence, separator in sentence_pieces(text):
            unknown = [n for n in self._names.findall(sentence) if self._invented(n, known)]
            if unknown:
                invented.append(sentence.strip())
            elif sentence.strip():
                kept.append(sentence.strip() + separator)
        return "".join(kept).strip(), invented


__all__ = ["AnswerGrounding", "GroundingOutcome", "Revise"]
