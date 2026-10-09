"""Embedding backends for the NOLLM cascade, RAG and memory scoring.

Two implementations behind one interface: a hosted embedding model (the configured
provider's), and
a deterministic local hashing embedder that needs no network and no model
download — the fallback that keeps tests fast, offline and reproducible.
"""

import hashlib
import math
import re
from typing import Literal, Protocol, runtime_checkable

from openai import AsyncOpenAI

from agent_platform.config import get_settings
from agent_platform.config.settings import LLMSettings
from agent_platform.llm.http import create_sdk_http_client
from agent_platform.logging import get_logger
from agent_platform.observability import checks
from agent_platform.resilience import external_api_error_fields

# The embedding backend that calls a provider (`llm.embedding.backend`); any other is local.
HOSTED = "hosted"
# The SDK requires a key even for an endpoint that needs none.
_NO_KEY = "not-needed"

_TOKEN_PATTERN = re.compile(r"[a-z0-9']+")

# What a text is, to a model that embeds questions and passages differently.
EmbeddingPurpose = Literal["query", "passage"]


@runtime_checkable
class EmbeddingClient(Protocol):
    """Turns text into vectors for similarity scoring."""

    async def embed(
        self, texts: list[str], purpose: EmbeddingPurpose = "query"
    ) -> list[list[float]]: ...

    @property
    def dimensions(self) -> int: ...

    @property
    def model_id(self) -> str:
        """The model that produced the most recent vectors.

        Vectors from two models live in different spaces. An index uses
        this to refuse to mix them — including when a hosted client has
        quietly fallen back to a local one.
        """
        ...


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Cosine similarity, clamped to [0, 1] for use as a score."""
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    if norm == 0:
        return 0.0
    return max(0.0, min(1.0, dot / norm))


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens — shared by the local embedder and BM25."""
    return _TOKEN_PATTERN.findall(text.lower())


class HashingEmbeddingClient:
    """Deterministic bag-of-words hashing embedder (no network, no weights).

    Token counts are hashed into a fixed number of buckets. Similar texts
    share buckets, so cosine similarity is a genuine lexical-overlap signal —
    enough for the cascade's cheap tier, and reproducible in tests.
    """

    def __init__(self, dimensions: int | None = None) -> None:
        self._dimensions = dimensions or get_settings().llm.embedding.local_dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_id(self) -> str:
        return f"{type(self).__name__}:{self._dimensions}"

    async def embed(
        self, texts: list[str], purpose: EmbeddingPurpose = "query"
    ) -> list[list[float]]:
        """Embed each text into a normalized hashed bag-of-words vector.

        Symmetric by construction, so `purpose` changes nothing here.
        """
        return [self.embed_sync(text) for text in texts]

    def embed_sync(self, text: str) -> list[float]:
        """Synchronous embedding — callers outside async paths use this."""
        vector = [0.0] * self._dimensions
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            vector[int.from_bytes(digest, "big") % self._dimensions] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


class HostedEmbeddingClient:
    """A hosted embedding model (`llm.embedding`), with the local embedder as a fallback.

    Entitlements are per-account, so a failure here degrades to the hashing
    embedder rather than taking down every caller that needs a similarity
    score.
    """

    def __init__(
        self,
        settings: LLMSettings | None = None,
        client: AsyncOpenAI | None = None,
        fallback: EmbeddingClient | None = None,
    ) -> None:
        self._settings = settings or get_settings().llm
        embedding = self._settings.embedding
        provider = self._settings.provider_settings(embedding.provider)
        self._model = embedding.model
        self._input_types = self._settings.embedding.input_types
        self._truncate = self._settings.embedding.truncate
        self._logger = get_logger("llm")
        self._fallback = fallback or HashingEmbeddingClient()
        self._dimensions = 0
        self._last_model_id = self._model
        self._client = client or AsyncOpenAI(
            base_url=provider.base_url,
            api_key=self._settings.api_key_for(embedding.provider) or _NO_KEY,
            timeout=provider.timeout_seconds,
            http_client=create_sdk_http_client(provider.timeout_seconds),
        )

    @property
    def dimensions(self) -> int:
        return self._dimensions or self._fallback.dimensions

    @property
    def model_id(self) -> str:
        return self._last_model_id

    async def embed(
        self, texts: list[str], purpose: EmbeddingPurpose = "query"
    ) -> list[list[float]]:
        """Embed via the hosted model, falling back locally if the endpoint refuses."""
        input_type = getattr(self._input_types, purpose)
        try:
            response = await self._client.embeddings.create(
                model=self._model,
                input=texts,
                extra_body={"input_type": input_type, "truncate": self._truncate},
            )
        except Exception as exc:
            self._logger.warning(
                "Hosted embeddings unavailable, using local fallback",
                error=str(exc),
                **(external_api_error_fields(exc) or {}),
            )
            self._last_model_id = self._fallback.model_id
            return await self._fallback.embed(texts, purpose)
        vectors = [item.embedding for item in response.data]
        self._dimensions = len(vectors[0]) if vectors else 0
        self._last_model_id = self._model
        return vectors


def _embedding_backend(client: object) -> str:
    """A hosted embedding model, or the local hashing embedder."""
    return checks.HOSTED_MODEL if isinstance(client, HostedEmbeddingClient) else checks.RULE


@checks.recorded_dependency(checks.EMBEDDING, "embed", _embedding_backend)
def create_embedding_client(settings: LLMSettings | None = None) -> EmbeddingClient:
    """Build the configured embedding client (Factory)."""
    resolved = settings or get_settings().llm
    if resolved.embedding.backend == HOSTED:
        return HostedEmbeddingClient(resolved)
    return HashingEmbeddingClient(resolved.embedding.local_dimensions)
