"""Reusable Hugging Face intent-classifier implementation."""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any

from agentic_commons.config import get_settings
from agentic_commons.intent.types import IntentCandidate, IntentClassifierInterface
from agentic_commons.logging import get_logger


class LocalModelArtifactUnavailableError(FileNotFoundError):
    """The repository-local model artifact is absent or structurally incomplete."""


class HuggingFaceIntentClassifier(IntentClassifierInterface):
    """Local Hugging Face sequence classifier with durable on-disk caching.

    The model is downloaded only when the configured local artifact is missing
    or cannot be loaded. Once loaded, the model remains resident in this object
    for the life of the agent process.
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
        self._model_name = model or settings.model
        self._local_model_dir = Path(local_model_dir or settings.resolved_local_model_dir)
        self._download_if_missing = (
            settings.download_if_missing if download_if_missing is None else download_if_missing
        )
        self._max_length = max_length or settings.max_length
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._torch: Any | None = None
        self._load_lock = asyncio.Lock()
        self._sync_load_lock = threading.Lock()
        self._logger = get_logger("llm")

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def score_type(self) -> str:
        return "model_softmax"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None and self._torch is not None

    @property
    def cache_path(self) -> Path:
        """Repository-local model artifact directory."""
        return self._local_model_dir

    async def ensure_loaded(self) -> None:
        """Load the cached model or download it once when the cache is absent."""
        if not self._settings.enabled:
            raise RuntimeError("intent classifier disabled")
        await self._get_model()

    def ensure_loaded_sync(self) -> None:
        """Synchronously load the model before a TUI/event loop owns file descriptors.

        The customer-support TUI starts an asyncio/Textual event loop and opens
        its own descriptors. On macOS, native ML/process machinery can trip
        Python's descriptor validation when initialization happens after that
        boundary. Preloading the model in the ordinary main thread keeps model
        initialization outside the TUI lifecycle.
        """
        if not self._settings.enabled:
            raise RuntimeError("intent classifier disabled")
        if self.is_loaded:
            return

        with self._sync_load_lock:
            if self.is_loaded:
                return
            self._preflight_dependencies()
            try:
                tokenizer, model, torch = self._load_local()
                self._logger.info(
                    "Intent classifier loaded from local cache",
                    model=self._model_name,
                    path=str(self._local_model_dir),
                )
            except LocalModelArtifactUnavailableError as local_error:
                if not self._download_if_missing:
                    raise RuntimeError(
                        f"Intent classifier local model unavailable at {self._local_model_dir}: {local_error}"
                    ) from local_error
                self._logger.warning(
                    "Intent classifier local artifact unavailable; downloading",
                    model=self._model_name,
                    path=str(self._local_model_dir),
                    reason=str(local_error),
                )
                tokenizer, model, torch = self._download_and_load()
                self._logger.info(
                    "Intent classifier downloaded and cached",
                    model=self._model_name,
                    path=str(self._local_model_dir),
                )

            self._tokenizer = tokenizer
            self._model = model
            self._torch = torch

    async def classify(self, text: str) -> list[IntentCandidate]:
        await self.ensure_loaded()

        tokenizer, model, torch = await self._get_model()

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

    async def _get_model(self) -> tuple[Any, Any, Any]:
        if self._model is not None and self._tokenizer is not None and self._torch is not None:
            return self._tokenizer, self._model, self._torch

        async with self._load_lock:
            if self._model is not None and self._tokenizer is not None and self._torch is not None:
                return self._tokenizer, self._model, self._torch

            try:
                tokenizer, model, torch = await asyncio.to_thread(self._load_local)
                self._logger.info(
                    "Intent classifier loaded from local cache",
                    model=self._model_name,
                    path=str(self._local_model_dir),
                )
            except LocalModelArtifactUnavailableError as local_error:
                if not self._download_if_missing:
                    raise RuntimeError(
                        f"Intent classifier local model unavailable at {self._local_model_dir}: {local_error}"
                    ) from local_error

                self._logger.warning(
                    "Intent classifier local artifact unavailable; downloading",
                    model=self._model_name,
                    path=str(self._local_model_dir),
                    reason=str(local_error),
                )
                tokenizer, model, torch = await asyncio.to_thread(self._download_and_load)
                self._logger.info(
                    "Intent classifier downloaded and cached",
                    model=self._model_name,
                    path=str(self._local_model_dir),
                )

            self._tokenizer = tokenizer
            self._model = model
            self._torch = torch
            return tokenizer, model, torch

    @staticmethod
    def _preflight_dependencies() -> None:
        """Fail before model work when required runtime packages are absent."""
        missing: list[str] = []
        for module_name, package_name in (
            ("torch", "torch"),
            ("transformers", "transformers"),
            ("huggingface_hub", "huggingface-hub"),
        ):
            try:
                __import__(module_name)
            except ModuleNotFoundError:
                missing.append(package_name)

        if missing:
            packages = " ".join(sorted(set(missing)))
            raise RuntimeError(
                "Intent classifier runtime dependencies are missing: "
                f"{packages}. Install the project dependencies with "
                "'./venv/bin/python -m pip install -e .' and retry."
            )

    def _load_local(self) -> tuple[Any, Any, Any]:
        if not self._local_model_dir.exists():
            raise LocalModelArtifactUnavailableError("local model directory does not exist")
        return self._load_local_from(self._local_model_dir)


    def _download_and_load(self) -> tuple[Any, Any, Any]:
        import torch
        from huggingface_hub import snapshot_download

        parent = self._local_model_dir.parent
        parent.mkdir(parents=True, exist_ok=True)
        temp_root = Path(
            tempfile.mkdtemp(prefix=f"{self._local_model_dir.name}.", dir=parent)
        )
        try:
            snapshot_download(
                repo_id=self._model_name,
                local_dir=str(temp_root),
                allow_patterns=[
                    "config.json",
                    "*.safetensors",
                    "pytorch_model*.bin",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "vocab.txt",
                    "merges.txt",
                    "added_tokens.json",
                ],
            )

            tokenizer, model, _ = self._load_local_from(temp_root)

            if self._local_model_dir.exists():
                shutil.rmtree(self._local_model_dir)
            temp_root.rename(self._local_model_dir)

            # Re-open from the final path so the resident objects are backed
            # by exactly the artifact we intend to reuse on subsequent runs.
            tokenizer, model, _ = self._load_local()
            model.eval()
            return tokenizer, model, torch
        except Exception:
            shutil.rmtree(temp_root, ignore_errors=True)
            raise

    def _load_local_from(self, model_dir: Path) -> tuple[Any, Any, Any]:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        weight_files = (
            list(model_dir.glob("*.safetensors"))
            + list(model_dir.glob("pytorch_model*.bin"))
        )
        if not (model_dir / "config.json").exists():
            raise LocalModelArtifactUnavailableError("model is missing config.json")
        if not weight_files:
            raise LocalModelArtifactUnavailableError("model is missing model weights")
        if not ((model_dir / "tokenizer.json").exists() or (model_dir / "vocab.txt").exists()):
            raise LocalModelArtifactUnavailableError("model is missing tokenizer artifacts")

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            local_files_only=True,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir),
            local_files_only=True,
        )
        model.eval()
        return tokenizer, model, torch



def create_intent_classifier() -> IntentClassifierInterface:
    """Create the configured shared intent-classifier provider."""
    settings = get_settings().llm.intent_classifier
    if settings.provider == "huggingface_local":
        return HuggingFaceIntentClassifier()
    raise RuntimeError(f"Unsupported intent classifier provider: {settings.provider}")
