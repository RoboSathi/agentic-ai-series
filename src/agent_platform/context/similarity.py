"""How alike two texts are — one seam for every check that asks.

Several checks need the same question answered: is this retrieved chunk about the task
(distraction), is it a copy of one already in the window (consolidation, confusion), are these
two sentences about the same thing (the clash shortlist), is this step the step that already
failed (the loop's semantic stagnation tier). Each used to answer it with its own word overlap.

`TextSimilarity` answers it once. The embedding implementation runs over whichever local
encoder the agent declares (`llm.local_embedder`), else the hashing embedder — never a hosted
endpoint, so a check never spends a rate-limited provider call. `LexicalSimilarity` (Jaccard
over word sets) is the floor every check can fall back to.
"""

from functools import cache
from typing import Protocol, runtime_checkable

from agent_platform.config import get_settings
from agent_platform.llm.embeddings import (
    EmbeddingClient,
    HashingEmbeddingClient,
    cosine_similarity,
    tokenize,
)
from agent_platform.logging import get_logger


@runtime_checkable
class TextSimilarity(Protocol):
    """Pairwise similarity in [0, 1] between every left text and every right text."""

    @property
    def name(self) -> str: ...

    async def matrix(self, left: list[str], right: list[str]) -> list[list[float]]: ...


class LexicalSimilarity:
    """Jaccard similarity of word sets: deterministic, free, and blind to paraphrase."""

    @property
    def name(self) -> str:
        return "lexical"

    async def matrix(self, left: list[str], right: list[str]) -> list[list[float]]:
        right_words = [set(tokenize(text)) for text in right]
        return [[jaccard_sets(set(tokenize(a)), b) for b in right_words] for a in left]


class EmbeddingSimilarity:
    """Cosine similarity of sentence vectors from a local encoder.

    Degrades rather than raises: an encoder that fails falls back to lexical similarity for
    the life of the process, and says so once.
    """

    def __init__(self, embedder: EmbeddingClient, fallback: TextSimilarity | None = None) -> None:
        self._embedder = embedder
        self._fallback = fallback or LexicalSimilarity()
        self._failed = False
        self._logger = get_logger("context")

    @property
    def name(self) -> str:
        return self._fallback.name if self._failed else self._embedder.model_id

    async def matrix(self, left: list[str], right: list[str]) -> list[list[float]]:
        if not left or not right:
            return [[] for _ in left]
        if self._failed:
            return await self._fallback.matrix(left, right)
        try:
            vectors = await self._embedder.embed([*left, *right], purpose="passage")
        except Exception as exc:
            self._failed = True
            self._logger.warning("Similarity encoder failed; using word overlap", error=str(exc))
            return await self._fallback.matrix(left, right)
        mine, theirs = vectors[: len(left)], vectors[len(left) :]
        return [[cosine_similarity(a, b) for b in theirs] for a in mine]


def jaccard(left: str, right: str) -> float:
    """Word-set overlap of two texts."""
    return jaccard_sets(set(tokenize(left)), set(tokenize(right)))


def jaccard_sets(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


@cache
def shared_similarity() -> TextSimilarity:
    """The process's similarity: the configured local encoder, else hashed words, else words."""
    backend = get_settings().context.similarity.backend
    if backend == "local_embedder" and get_settings().llm.local_embedder.enabled:
        from agent_platform.llm.local_embeddings import LocalSentenceEmbedder

        return EmbeddingSimilarity(LocalSentenceEmbedder())
    if backend == "lexical":
        return LexicalSimilarity()
    return EmbeddingSimilarity(HashingEmbeddingClient())


__all__ = [
    "EmbeddingSimilarity",
    "LexicalSimilarity",
    "TextSimilarity",
    "jaccard",
    "jaccard_sets",
    "shared_similarity",
]
