"""The embedding cache: each (purpose, text) embedded once, never two spaces in one answer."""

from agent_platform.config import get_settings
from agent_platform.llm.embedding_cache import CachingEmbeddingClient, cached
from agent_platform.llm.embeddings import EmbeddingPurpose


class _Recording:
    """An embedder that records every text it was asked for, under a switchable model."""

    def __init__(self) -> None:
        self.asked: list[tuple[str, str]] = []
        self.model = "space-a"

    @property
    def model_id(self) -> str:
        return self.model

    @property
    def dimensions(self) -> int:
        return 2

    async def embed(
        self, texts: list[str], purpose: EmbeddingPurpose = "query"
    ) -> list[list[float]]:
        self.asked += [(purpose, text) for text in texts]
        return [[float(len(text)), 1.0 if self.model == "space-a" else 2.0] for text in texts]


def _cache(inner: _Recording, max_entries: int | None = None) -> CachingEmbeddingClient:
    settings = get_settings().llm.embedding.cache
    if max_entries is not None:
        settings = settings.model_copy(update={"max_entries": max_entries})
    return CachingEmbeddingClient(inner, settings)


async def test_a_text_seen_before_is_not_embedded_again() -> None:
    inner = _Recording()
    cache = _cache(inner)
    first = await cache.embed(["a", "bb"], "passage")
    assert await cache.embed(["bb", "a", "ccc"], "passage") == [first[1], first[0], [3.0, 1.0]]
    assert inner.asked == [("passage", "a"), ("passage", "bb"), ("passage", "ccc")]


async def test_the_purpose_is_part_of_the_key() -> None:
    inner = _Recording()
    cache = _cache(inner)
    await cache.embed(["a"], "query")
    await cache.embed(["a"], "passage")
    assert inner.asked == [("query", "a"), ("passage", "a")]


async def test_a_repeated_text_in_one_batch_is_embedded_once() -> None:
    inner = _Recording()
    assert len(await _cache(inner).embed(["a", "a", "a"])) == 3
    assert inner.asked == [("query", "a")]


async def test_a_change_of_model_never_mixes_two_spaces() -> None:
    inner = _Recording()
    cache = _cache(inner)
    await cache.embed(["a"])
    inner.model = "space-b"
    vectors = await cache.embed(["a", "bb"])
    assert [vector[1] for vector in vectors] == [2.0, 2.0]


async def test_a_batch_larger_than_the_bound_is_still_answered_whole() -> None:
    inner = _Recording()
    cache = _cache(inner, max_entries=2)
    await cache.embed(["a"])
    assert len(await cache.embed(["a", "bb", "ccc", "dddd"])) == 4


async def test_the_least_recently_used_goes_first() -> None:
    inner = _Recording()
    cache = _cache(inner, max_entries=2)
    await cache.embed(["a", "bb"])
    await cache.embed(["a"])
    await cache.embed(["ccc"])
    inner.asked.clear()
    await cache.embed(["a", "bb"])
    assert inner.asked == [("query", "bb")]


def test_off_means_the_client_itself() -> None:
    inner = _Recording()
    settings = get_settings().llm.embedding.cache.model_copy(update={"enabled": False})
    assert cached(inner, settings) is inner
