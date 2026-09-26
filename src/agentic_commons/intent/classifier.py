"""Reusable Hugging Face intent-classifier implementation."""

from __future__ import annotations

import asyncio
from pathlib import Path

from agentic_commons.config import get_settings
from agentic_commons.intent.types import IntentCandidate, IntentClassifierInterface
from agentic_commons.llm.local_models import LocalSequenceClassifier


class HuggingFaceIntentClassifier(IntentClassifierInterface):
    """Local Hugging Face sequence classifier with durable on-disk caching.

    The model is downloaded only when the configured local artifact is missing
    or cannot be loaded. Once loaded, the model remains resident in this object
    for the life of the agent process. The lifecycle itself is
    `LocalSequenceClassifier`'s; this class only turns logits into intents.
    """

    def __init__(
        self,
        model: str | None = None,
        local_model_dir: str | Path | None = None,
        download_if_missing: bool | None = None,
        max_length: int | None = None,
    ) -> None:
        settings = get_settings().llm.intent_classifier
        self._settings = settings
        self._artifact = LocalSequenceClassifier(
            model_name=model or settings.model,
            local_dir=Path(local_model_dir or settings.resolved_local_model_dir),
            download_if_missing=(
                settings.download_if_missing if download_if_missing is None else download_if_missing
            ),
            purpose="Intent classifier",
        )
        self._max_length = max_length or settings.max_length

    @property
    def model_name(self) -> str:
        return self._artifact.model_name

    @property
    def score_type(self) -> str:
        return "model_softmax"

    @property
    def is_loaded(self) -> bool:
        return self._artifact.is_loaded

    @property
    def cache_path(self) -> Path:
        """Repository-local model artifact directory."""
        return self._artifact.local_dir

    async def ensure_loaded(self) -> None:
        """Load the cached model or download it once when the cache is absent."""
        self._require_enabled()
        await self._artifact.load()

    def ensure_loaded_sync(self) -> None:
        """Synchronously load the model before a TUI/event loop owns file descriptors.

        The customer-support TUI starts an asyncio/Textual event loop and opens
        its own descriptors. On macOS, native ML/process machinery can trip
        Python's descriptor validation when initialization happens after that
        boundary. Preloading the model in the ordinary main thread keeps model
        initialization outside the TUI lifecycle.
        """
        self._require_enabled()
        self._artifact.load_sync()

    async def classify(self, text: str) -> list[IntentCandidate]:
        self._require_enabled()
        tokenizer, model, torch = await self._artifact.load()

        def _predict() -> list[IntentCandidate]:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self._max_length,
            )
            with torch.no_grad():
                logits = model(**inputs).logits[0]
                probabilities = torch.softmax(logits, dim=-1).tolist()

            id2label = getattr(model.config, "id2label", {}) or {}
            candidates = [
                IntentCandidate(
                    intent=str(id2label.get(index, str(index))),
                    score=float(score),
                )
                for index, score in enumerate(probabilities)
            ]
            candidates.sort(key=lambda item: item.score, reverse=True)
            return candidates

        return await asyncio.to_thread(_predict)

    def _require_enabled(self) -> None:
        if not self._settings.enabled:
            raise RuntimeError("intent classifier disabled")


def create_intent_classifier() -> IntentClassifierInterface:
    """Create the configured shared intent-classifier provider."""
    settings = get_settings().llm.intent_classifier
    if settings.provider == "huggingface_local":
        return HuggingFaceIntentClassifier()
    raise RuntimeError(f"Unsupported intent classifier provider: {settings.provider}")
