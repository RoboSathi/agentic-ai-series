"""An embedding client that remembers what it embedded (Decorator).

Ranking embeds the same candidates every turn: the offered tools' descriptions, the knowledge
base's passages, the evidence a loop has gathered. Each was a hosted request, or a local
encoder pass, for a vector the process already had. This wraps any `EmbeddingClient` with a
bounded LRU keyed by purpose and the text's digest, so only text never seen is embedded.

Vectors from two models live in different spaces, so the cache holds one model's vectors at a
time: when the inner client answers from another model (a hosted client that fell back to the
local one), the cache is emptied, and a batch is never answered half from each space.
"""

import hashlib
from collections import OrderedDict
from typing import TYPE_CHECKING

from agent_platform.config.settings import EmbeddingCacheSettings

if TYPE_CHECKING:
    # Types only: `embeddings` builds its clients behind this cache, so it imports this module.
    from agent_platform.llm.embeddings import EmbeddingClient, EmbeddingPurpose

_Key = tuple[str, str]


class CachingEmbeddingClient:
    """An `EmbeddingClient` that embeds each (purpose, text) once, up to `max_entries`."""

    def __init__(self, inner: "EmbeddingClient", settings: EmbeddingCacheSettings) -> None:
        self._inner = inner
        self._max_entries = settings.max_entries
        self._vectors: OrderedDict[_Key, list[float]] = OrderedDict()
        self._model = inner.model_id

    @property
    def inner(self) -> "EmbeddingClient":
        """The client this one decorates."""
        return self._inner

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    async def embed(
        self, texts: list[str], purpose: "EmbeddingPurpose" = "query"
    ) -> list[list[float]]:
        """Cached vectors where there are any; the rest embedded in one call."""
        keys = [(purpose, _digest(text)) for text in texts]
        known = {key: self._recall(key) for key in set(keys) if key in self._vectors}
        missing = list(dict.fromkeys(key for key in keys if key not in known))
        if not missing:
            return [known[key] for key in keys]
        by_key = dict(zip(keys, texts))
        fresh = dict(zip(missing, await self._inner.embed([by_key[k] for k in missing], purpose)))
        if self._inner.model_id != self._model:
            self._reset(self._inner.model_id)
            if known:
                # The hits are from the old space: answer the whole batch from the new one.
                return await self._inner.embed(texts, purpose)
        for key, vector in fresh.items():
            self._store(key, vector)
        return [known[key] if key in known else fresh[key] for key in keys]

    def _recall(self, key: _Key) -> list[float]:
        self._vectors.move_to_end(key)
        return self._vectors[key]

    def _store(self, key: _Key, vector: list[float]) -> None:
        self._vectors[key] = vector
        self._vectors.move_to_end(key)
        while len(self._vectors) > self._max_entries:
            self._vectors.popitem(last=False)

    def _reset(self, model: str) -> None:
        self._vectors.clear()
        self._model = model


def cached(client: "EmbeddingClient", settings: EmbeddingCacheSettings) -> "EmbeddingClient":
    """The client behind a cache when the cache is on; the client itself when it is off."""
    return CachingEmbeddingClient(client, settings) if settings.enabled else client


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = ["CachingEmbeddingClient", "cached"]
