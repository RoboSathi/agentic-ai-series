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
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from agent_platform.config import get_settings
from agent_platform.config.settings import IntentClassifierSettings
from agent_platform.intent.types import IntentCandidate
from agent_platform.llm.local_laya import (
    Fetcher,
    Loader,
    checkpoint_folder,
    fetch_checkpoint,
    load_laya,
)
from agent_platform.logging import get_logger

_SCORE_TYPE = "laya_calibrated"
_MIN_LABELS = 2


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
        self._loader = loader or load_laya
        self._fetcher = fetcher or fetch_checkpoint
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
        return checkpoint_folder(
            self.cache_path,
            self._laya.model,
            self._laya.subfolder,
            download=self._settings.download_if_missing,
            fetcher=self._fetcher,
        )

    def _require_enabled(self) -> None:
        if not self._settings.enabled:
            raise RuntimeError("intent classifier disabled")
