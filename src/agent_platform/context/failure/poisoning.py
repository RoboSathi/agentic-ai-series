"""Poisoning: something false or hostile in the window that the model would take as given.

Two ways in. An instruction smuggled into data (a retrieved page that says "ignore your
rules") — found by the security layer's phrase screen. And a claim nobody verified that a
verified source contradicts: most often the model's own earlier answer, carried forward in the
transcript, saying a record changed when the record read this turn says it did not.
The first drops the chunk. The second cuts only the offending line, so the rest of the
transcript stays.
"""

from typing import Protocol, runtime_checkable

from agent_platform.config import get_settings
from agent_platform.context.failure.common import (
    ContradictionModel,
    body,
    contradicts,
    finding,
    label,
    same_subject,
    sentences,
)
from agent_platform.context.similarity import TextSimilarity
from agent_platform.context.trust import rank, trust_of
from agent_platform.context.types import AssembledContext, Chunk, ContextFinding

# Kinds that carry data from outside the instruction boundary — where an
# injected instruction would be poisoning rather than a request.
_DATA_KINDS = frozenset({"retrieved", "tool_result", "memory", "history"})


@runtime_checkable
class InjectionScreen(Protocol):
    """Finds injection phrasing in text (the security layer's separator satisfies this)."""

    def find_phrases(self, text: str) -> list[str]: ...


class PoisoningCheck:
    def __init__(
        self,
        screen: InjectionScreen | None,
        nli: ContradictionModel | None,
        similarity: TextSimilarity,
    ) -> None:
        self._screen = screen
        self._nli = nli
        self._similarity = similarity

    async def findings(self, context: AssembledContext) -> list[ContextFinding]:
        return [*self._injected(context), *await self._unverified(context)]

    def _injected(self, context: AssembledContext) -> list[ContextFinding]:
        if self._screen is None:
            return []
        found = []
        for index, chunk in enumerate(context.chunks):
            phrases = self._screen.find_phrases(chunk.content) if chunk.kind in _DATA_KINDS else []
            if phrases:
                found.append(
                    finding("poisoning", "poisoning_injection", source=label(chunk),
                            action="dropped", index=index, phrases=phrases)
                )  # fmt: skip
        return found

    async def _unverified(self, context: AssembledContext) -> list[ContextFinding]:
        """The model's own earlier claims that a verified chunk contradicts."""
        if self._nli is None or not get_settings().context.failure_detection.nli:
            return []
        facts = [(c, s) for c in context.chunks if _verified(c) for s in sentences(body(c))]
        found: list[ContextFinding] = []
        for index, chunk in enumerate(context.chunks):
            claims = _model_lines(chunk)
            if facts and claims:
                found.extend(await self._against(index, chunk, claims, facts))
        return found

    async def _against(
        self, index: int, chunk: Chunk, claims: list[str], facts: list[tuple[Chunk, str]]
    ) -> list[ContextFinding]:
        limit = get_settings().context.failure_detection.max_pairs
        texts = [fact for _, fact in facts]
        similar = await self._similarity.matrix(claims, texts)
        found: list[ContextFinding] = []
        for claim, row in zip(claims, similar):
            # Only facts about the record the claim names; the most similar first, capped.
            ranked = sorted(
                ((score, pair) for pair, score in zip(facts, row) if same_subject(claim, pair[1])),
                key=lambda item: item[0],
                reverse=True,
            )
            for _, (source, fact) in ranked[:limit]:
                result = await self._nli.entails(fact, claim)  # type: ignore[union-attr]
                if contradicts(result):
                    found.append(
                        finding("poisoning", "poisoning_claim", source=label(chunk),
                                action="excised", index=index, excise=[claim],
                                statements=[claim, fact], score=result.score, against=label(source))
                    )  # fmt: skip
                    break
        return found


def _verified(chunk: Chunk) -> bool:
    """A source the model's word is checked against: verified records and the harness itself."""
    return rank(trust_of(chunk)) >= rank("verified")


def _model_lines(chunk: Chunk) -> list[str]:
    """The lines of a transcript the model wrote — claims no one has verified."""
    if chunk.kind != "history":
        return []
    prefix = get_settings().context.trust.model_line_prefix
    lines = [line.strip() for line in body(chunk).splitlines()]
    return [
        claim
        for line in lines
        if line.casefold().startswith(prefix.casefold())
        for claim in sentences(line[len(prefix) :])
    ]
