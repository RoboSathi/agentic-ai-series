"""Sentence vectors from a local Hugging Face encoder — similarity with no network call.

The hosted embedder answers every text with a request: fine for a knowledge base indexed once,
a cost on every message for a classifier that embeds each one. A small sentence-embedding model
(MiniLM, bge, e5) runs in-process in milliseconds. Vectors are the encoder's last hidden states
averaged over the real tokens and normalised to unit length — what sentence-transformers does —
so cosine similarity is a dot product.

Loaded and downloaded through the same loader as every local model (`local_models.py`), so it
is warmed before a TUI starts and fetched by `make models` on a fresh checkout.
"""

import asyncio
from typing import Any

from agent_platform.config import get_settings
from agent_platform.config.settings import LocalEmbedderSettings
from agent_platform.llm.embeddings import EmbeddingPurpose
from agent_platform.llm.local_models import LocalSequenceClassifier

# The encoder without a task head.
_ENCODER = "AutoModel"


class LocalSentenceEmbedder:
    """An `EmbeddingClient` over a local encoder: mean-pooled, unit-length vectors."""

    def __init__(self, settings: LocalEmbedderSettings | None = None) -> None:
        self._settings = settings or get_settings().llm.local_embedder
        self._artifact = LocalSequenceClassifier(
            model_name=self._settings.model,
            local_dir=self._settings.resolved_local_model_dir,
            download_if_missing=self._settings.download_if_missing,
            purpose="sentence embedder",
            auto_class=_ENCODER,
        )

    @property
    def model_id(self) -> str:
        return self._settings.model

    @property
    def dimensions(self) -> int:
        """The encoder's hidden size: the length of every vector it returns."""
        _, model, _ = self._artifact.load_sync()
        return int(model.config.hidden_size)

    def ensure_loaded_sync(self) -> None:
        """Load (or download) now, in this thread — before a TUI owns the descriptors."""
        self._artifact.load_sync()

    async def embed(
        self, texts: list[str], purpose: EmbeddingPurpose = "query"
    ) -> list[list[float]]:
        if not texts:
            return []
        tokenizer, model, torch = await self._artifact.load()
        prefix = self._settings.prefixes.get(purpose, "")
        return await asyncio.to_thread(
            self._encode, tokenizer, model, torch, [f"{prefix}{text}" for text in texts]
        )

    def _encode(
        self, tokenizer: Any, model: Any, torch: Any, texts: list[str]
    ) -> list[list[float]]:
        batch = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self._settings.max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            hidden = model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        vectors = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return [[float(value) for value in row] for row in vectors.tolist()]


__all__ = ["LocalSentenceEmbedder"]
