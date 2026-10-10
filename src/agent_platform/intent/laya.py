"""Intent classification as one Laya choice question: the agent's labels are the options.

A fine-tuned sequence classifier answers in the labels it was trained on, and an
agent maps those onto its own; a label it has no mapping for cannot be reached.
A Laya decision model reads the options' descriptions instead, so the labels are
whatever the agent injects — adding one is a config edit, not a retraining.

Like every local model here, the checkpoint lives inside the repository: downloaded
once into its folder, loaded from that folder ever after, resident for the process.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import threading
from collections.abc import Callable, Mapping
from functools import cache
from pathlib import Path
from typing import Any

from agent_platform.config import get_settings
from agent_platform.config.settings import IntentClassifierSettings
from agent_platform.intent.types import IntentCandidate
from agent_platform.logging import get_logger

# Files a Laya checkpoint folder cannot load without.
_REQUIRED_FILES = ("model.safetensors", "rl_agent_config.json")
_INSTALL = "make install"
_SCORE_TYPE = "laya_calibrated"
_MIN_LABELS = 2

# (checkpoint folder, device) → a loaded Laya agent.
Loader = Callable[[Path, str], Any]
# (repository, subfolder, destination folder) → the checkpoint written there.
Fetcher = Callable[[str, str, Path], None]


class LayaIntentClassifier:
    """Asks a Laya checkpoint which of the injected labels a message is."""

    def __init__(
        self,
        choices: Mapping[str, str],
        instructions: str,
        settings: IntentClassifierSettings | None = None,
        loader: Loader | None = None,
        fetcher: Fetcher | None = None,
    ) -> None:
        if len(choices) < _MIN_LABELS:
            raise ValueError("a choice question needs at least two labels")
        self._settings = settings or get_settings().llm.intent_classifier
        self._laya = self._settings.laya
        self._question = {
            self._laya.question_id: {
                "type": "choice",
                "instructions": instructions,
                "criteria": dict(choices),
            }
        }
        self._loader = loader or _load_laya
        self._fetcher = fetcher or _fetch_checkpoint
        self._agent: Any = None
        self._lock = threading.Lock()
        self._logger = get_logger("intent")

    @property
    def model_name(self) -> str:
        return f"{self._laya.model}/{self._laya.subfolder}"

    @property
    def score_type(self) -> str:
        return _SCORE_TYPE

    @property
    def is_loaded(self) -> bool:
        return self._agent is not None

    @property
    def cache_path(self) -> Path:
        """Repository-local checkpoint folder."""
        return self._laya.resolved_local_model_dir

    async def ensure_loaded(self) -> None:
        """Load the checkpoint, downloading it once when the folder is missing."""
        await asyncio.to_thread(self.ensure_loaded_sync)

    def ensure_loaded_sync(self) -> None:
        """Load now, in this thread, before an event loop owns the process."""
        self._require_enabled()
        with self._lock:
            if self._agent is None:
                self._agent = self._loader(self._artifact(), self._laya.device)
                self._logger.info("Laya intent classifier loaded", model=self.model_name)

    async def classify(self, text: str) -> list[IntentCandidate]:
        """Every injected label with its probability, best first."""
        self._require_enabled()
        await self.ensure_loaded()
        result = await asyncio.to_thread(
            self._agent.system_one, {self._laya.state_key: text}, self._question
        )
        probabilities = result["answers"][self._laya.question_id]["probabilities"]
        candidates = [
            IntentCandidate(intent=label, score=float(score))
            for label, score in probabilities.items()
        ]
        return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)

    def _artifact(self) -> Path:
        """The checkpoint folder, fetched first if it is missing and fetching is allowed."""
        folder = self.cache_path
        if _complete(folder):
            return folder
        if not self._settings.download_if_missing:
            raise RuntimeError(f"Laya checkpoint unavailable at {folder}")
        self._logger.warning("Laya checkpoint missing; downloading", model=self.model_name)
        self._fetcher(self._laya.model, self._laya.subfolder, folder)
        return folder

    def _require_enabled(self) -> None:
        if not self._settings.enabled:
            raise RuntimeError("intent classifier disabled")


def _complete(folder: Path) -> bool:
    return all((folder / name).exists() for name in _REQUIRED_FILES)


def _load_laya(folder: Path, device: str) -> Any:
    """The checkpoint in this folder, loaded once per process however many routers ask.

    Every `/harness` rebuild builds a new router; reloading ~800 MB of weights
    for each would cost seconds per toggle, for weights that never change.
    """
    return _resident(str(folder), device)


@cache
def _resident(folder: str, device: str) -> Any:
    try:
        import laya
    except ImportError as missing:
        raise RuntimeError(f"Laya is not installed: run `{_INSTALL}`") from missing
    return laya.load(folder, device=device)


def _fetch_checkpoint(repository: str, subfolder: str, destination: Path) -> None:
    """Download one checkpoint of a bundle into a sibling folder, then rename it into place.

    A download that dies halfway leaves the staging folder, never a half-filled
    checkpoint folder that the next start would try to load.
    """
    from huggingface_hub import snapshot_download

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"{destination.name}.", dir=destination.parent))
    try:
        snapshot_download(
            repo_id=repository, local_dir=str(staging), allow_patterns=[f"{subfolder}/*"]
        )
        if destination.exists():
            shutil.rmtree(destination)
        (staging / subfolder).rename(destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
