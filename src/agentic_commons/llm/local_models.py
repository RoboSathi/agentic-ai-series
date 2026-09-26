"""Repository-local Hugging Face sequence-classification models.

Every local encoder the harness runs — the intent classifier, the NLI
cross-encoder — has the same lifecycle: load from a folder inside the
repository, download once into that folder when it is missing, then stay
resident for the life of the process. This module owns that lifecycle so
each model is a thin adapter over it rather than a copy of it.

Files live in the repository, not the user's global Hugging Face cache,
and are opened with `local_files_only=True`: a model that loaded yesterday
loads today with the network down.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any

from agentic_commons.logging import get_logger

# What a sequence-classification checkpoint needs on disk, and nothing more.
_ARTIFACT_PATTERNS = [
    "config.json",
    "*.safetensors",
    "pytorch_model*.bin",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "spm.model",
    "vocab.txt",
    "merges.txt",
    "added_tokens.json",
]
# Tokenizer files, any one of which is enough to rebuild the tokenizer.
_TOKENIZER_FILES = ("tokenizer.json", "vocab.txt", "spm.model")
# Import name → pip package name, for a missing-dependency message.
_RUNTIME_DEPENDENCIES = (
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("huggingface_hub", "huggingface-hub"),
)

LoadedModel = tuple[Any, Any, Any]


class LocalModelArtifactUnavailableError(FileNotFoundError):
    """The repository-local model artifact is absent or structurally incomplete."""


class LocalSequenceClassifier:
    """One Hugging Face sequence classifier, resident once loaded.

    `purpose` names the model in every log line ("intent classifier",
    "NLI model"), because a warning that a model was downloaded is only
    useful if it says which one.
    """

    def __init__(
        self,
        model_name: str,
        local_dir: Path,
        download_if_missing: bool,
        purpose: str,
    ) -> None:
        self._model_name = model_name
        self._local_dir = local_dir
        self._download_if_missing = download_if_missing
        self._purpose = purpose
        self._loaded: LoadedModel | None = None
        self._async_lock = asyncio.Lock()
        self._sync_lock = threading.Lock()
        self._logger = get_logger("llm")

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def local_dir(self) -> Path:
        return self._local_dir

    @property
    def is_loaded(self) -> bool:
        return self._loaded is not None

    def load_sync(self) -> LoadedModel:
        """Load in the calling thread — before a TUI owns the descriptors on macOS."""
        if self._loaded is not None:
            return self._loaded
        with self._sync_lock:
            if self._loaded is None:
                self._loaded = self._load_or_download()
        return self._loaded

    async def load(self) -> LoadedModel:
        """Load off the event loop, once, however many callers ask."""
        if self._loaded is not None:
            return self._loaded
        async with self._async_lock:
            if self._loaded is None:
                self._loaded = await asyncio.to_thread(self.load_sync)
        return self._loaded

    def _load_or_download(self) -> LoadedModel:
        _preflight_dependencies(self._purpose)
        try:
            loaded = _load_from(self._local_dir)
            self._logger.info(f"{self._purpose} loaded from local cache", **self._where())
            return loaded
        except LocalModelArtifactUnavailableError as missing:
            if not self._download_if_missing:
                raise RuntimeError(
                    f"{self._purpose} local model unavailable at {self._local_dir}: {missing}"
                ) from missing
            self._logger.warning(
                f"{self._purpose} local artifact unavailable; downloading",
                reason=str(missing),
                **self._where(),
            )
            loaded = self._download()
            self._logger.info(f"{self._purpose} downloaded and cached", **self._where())
            return loaded

    def _download(self) -> LoadedModel:
        """Download into a sibling temp folder, then rename into place.

        A download that dies halfway leaves the temp folder, never a
        half-populated model folder that the next start would try to load.
        """
        from huggingface_hub import snapshot_download

        parent = self._local_dir.parent
        parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f"{self._local_dir.name}.", dir=parent))
        try:
            snapshot_download(
                repo_id=self._model_name,
                local_dir=str(staging),
                allow_patterns=_ARTIFACT_PATTERNS,
            )
            _load_from(staging)
            if self._local_dir.exists():
                shutil.rmtree(self._local_dir)
            staging.rename(self._local_dir)
            # Re-open from the final path, so the resident model is backed by
            # exactly the artifact the next start will reuse.
            return _load_from(self._local_dir)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def _where(self) -> dict[str, str]:
        return {"model": self._model_name, "path": str(self._local_dir)}


def _preflight_dependencies(purpose: str) -> None:
    """Fail before any model work when a runtime package is absent."""
    missing = []
    for module_name, package_name in _RUNTIME_DEPENDENCIES:
        try:
            __import__(module_name)
        except ModuleNotFoundError:
            missing.append(package_name)
    if missing:
        raise RuntimeError(
            f"{purpose} runtime dependencies are missing: {' '.join(sorted(set(missing)))}. "
            "Install the project dependencies and retry."
        )


def _load_from(model_dir: Path) -> LoadedModel:
    """Tokenizer, model and torch from one folder, local files only."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    _require_artifact(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), local_files_only=True)
    model.eval()
    return tokenizer, model, torch


def _require_artifact(model_dir: Path) -> None:
    """Refuse a folder that is missing its config, weights or tokenizer."""
    if not model_dir.exists():
        raise LocalModelArtifactUnavailableError("local model directory does not exist")
    if not (model_dir / "config.json").exists():
        raise LocalModelArtifactUnavailableError("model is missing config.json")
    if not (list(model_dir.glob("*.safetensors")) or list(model_dir.glob("pytorch_model*.bin"))):
        raise LocalModelArtifactUnavailableError("model is missing model weights")
    if not any((model_dir / name).exists() for name in _TOKENIZER_FILES):
        raise LocalModelArtifactUnavailableError("model is missing tokenizer artifacts")
