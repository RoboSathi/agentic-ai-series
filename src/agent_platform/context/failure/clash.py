"""Clash: two things in the window that cannot both be true.

Two detectors over the chunks that state facts. Structured `key: value` claims keyed by the
entity they are about (a stale retrieved copy of an order against this turn's read), cheap and
exact. And prose: sentence pairs from chunks of *different* trust, shortlisted by similarity
so only statements about the same thing are compared, then judged by the local NLI model.
Either way the more trusted side is kept and the other dropped; an equal-trust field clash is
flagged and never decided — there is no recency vote. Prose between equally trusted chunks is
not compared at all: NLI reads two conditional rules ("before it ships…", "once it ships…") as
a contradiction, and a tie it could not act on would only be noise on the card.
"""

from itertools import islice
from typing import Any

from agent_platform.config import get_settings
from agent_platform.context.failure.common import (
    ContradictionModel,
    body,
    contradicts,
    entities,
    finding,
    label,
    origin,
    same_subject,
    sentences,
)
from agent_platform.context.similarity import TextSimilarity
from agent_platform.context.trust import chunk_rank
from agent_platform.context.types import AssembledContext, Chunk, ContextFinding

# The chunk kinds that state facts about the world, as opposed to recounting what was said.
_ASSERTING_KINDS = frozenset({"policy", "retrieved", "tool_result", "memory"})

# A claim as stated: its value, and the chunk (with its position) that stated it.
_Stated = list[tuple[str, tuple[int, Chunk]]]


class ClashCheck:
    def __init__(self, nli: ContradictionModel | None, similarity: TextSimilarity) -> None:
        self._app_config = get_settings()
        self._nli = nli
        self._similarity = similarity

    async def findings(self, context: AssembledContext) -> list[ContextFinding]:
        return [*self._fields(context), *await self._prose(context)]

    def contradictions(self, context: AssembledContext) -> list[tuple[str, str]]:
        """Claim keys stated with different values, as `(key, "a vs b")`."""
        return [
            (key, " vs ".join(sorted({value for value, _ in stated})))
            for key, stated in claims_by_key(context).items()
            if len({value for value, _ in stated}) > 1
        ]

    def _fields(self, context: AssembledContext) -> list[ContextFinding]:
        found: list[ContextFinding] = []
        for key, stated in claims_by_key(context).items():
            if len({value for value, _ in stated}) > 1:
                lines = [f"{key}: {value} ({label(chunk)})" for value, (_, chunk) in stated]
                found += _resolve([(i, c) for _, (i, c) in stated], "clash_field",
                                  statements=lines, key=key)  # fmt: skip
        return found

    async def _prose(self, context: AssembledContext) -> list[ContextFinding]:
        settings = self._app_config.context.failure_detection
        if self._nli is None or not settings.nli:
            return []
        stated = list(islice(_statements(context), settings.max_sentences))
        found: list[ContextFinding] = []
        for (left, i, a), (right, j, b) in await self._pairs(stated):
            result = await self._nli.entails(a, b)
            if contradicts(result):
                found += _resolve([(i, left), (j, right)], "clash_prose",
                                  statements=[a, b], score=result.score,
                                  statement_sources=[origin(left), origin(right)])  # fmt: skip
        return found

    async def _pairs(
        self, stated: list[tuple[Chunk, int, str]]
    ) -> list[tuple[tuple[Chunk, int, str], tuple[Chunk, int, str]]]:
        """Sentence pairs from chunks of different trust that name the same record, the most
        similar first, at most `max_pairs` of them."""
        settings = self._app_config.context.failure_detection
        texts = [text for _, _, text in stated]
        similar = await self._similarity.matrix(texts, texts)
        ranked = sorted(
            (
                (similar[a][b], stated[a], stated[b])
                for a in range(len(stated))
                for b in range(a + 1, len(stated))
                if _decidable(stated[a][0], stated[b][0])
                and same_subject(stated[a][2], stated[b][2])
            ),
            key=lambda pair: pair[0],
            reverse=True,
        )
        return [(left, right) for _, left, right in ranked[: settings.max_pairs]]


def _decidable(left: Chunk, right: Chunk) -> bool:
    """Two chunks whose disagreement the window could act on: one is trusted more."""
    return left is not right and chunk_rank(left) != chunk_rank(right)


def _statements(context: AssembledContext) -> list[tuple[Chunk, int, str]]:
    return [
        (chunk, index, sentence)
        for index, chunk in enumerate(context.chunks)
        if chunk.kind in _ASSERTING_KINDS
        for sentence in sentences(body(chunk))
        if ":" not in sentence  # a `key: value` line is the structured detector's
    ]


def _resolve(
    stated: list[tuple[int, Chunk]],
    template: str,
    statements: list[str] | None = None,
    statement_sources: list[str] | None = None,
    **values: Any,
) -> list[ContextFinding]:
    """Keep the most trusted side of a clash; drop the rest, or flag a tie."""
    top = max(chunk_rank(chunk) for _, chunk in stated)
    tie = sum(chunk_rank(chunk) == top for _, chunk in stated) > 1
    return [
        finding(
            "clash",
            template,
            source=label(chunk),
            index=index,
            action="tie" if tie else "kept_higher" if chunk_rank(chunk) == top else "dropped_lower",
            statements=statements,
            statement_sources=statement_sources,
            **values,
        )
        for index, chunk in stated
    ]


def claims_by_key(context: AssembledContext) -> dict[str, _Stated]:
    """Every entity-keyed `key: value` claim in the asserting chunks."""
    claims: dict[str, _Stated] = {}
    for index, chunk in enumerate(context.chunks):
        if chunk.kind not in _ASSERTING_KINDS - {"memory"}:
            continue
        entity = _entity(chunk)
        for key, value in _claims(chunk):
            keyed = f"{entity} · {key}" if entity else key
            claims.setdefault(keyed, []).append((value, (index, chunk)))
    return claims


def _claims(chunk: Chunk) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for line in body(chunk).splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() and value.strip():
            found.append((key.strip().casefold(), value.strip().casefold()))
    return found


def _entity(chunk: Chunk) -> str:
    named = sorted(entities(body(chunk)), key=body(chunk).casefold().find)
    return named[0] if named else ""
