"""A second opinion on a claim NLI left undecided: it may confirm, never refute.

NLI compares a claim with one premise. A web page states one fact across several short lines
("Kumar Karan" / "Engineering Manager — AI @ Turing" / "Lead Instructor @ Scaler Academy"),
and a sentence that joins them is entailed by no single premise: NLI says *neutral*, rightly,
and a true claim stays unverified. A decision model reads the claim beside the source's own
text and answers one choice question — supported, contradicted, or not stated.

It only ever *confirms*. Measured on the RoboSathi turn it confirmed 5/5 true claims where NLI
confirmed 2/5, but it called a false claim "not stated" where NLI said "contradicted", and both
accepted a wrong relation ("Lead Instructor at Turing"). So a contradiction, a removal, a
repair are never its to decide: those stay with NLI and the deterministic checks.

    ClaimConfirmer        the seam: (claim, evidence text) → Confirmation
    LayaClaimConfirmer    a local Laya decision checkpoint (`llm/local_laya.py`)
    create_claim_confirmer  None unless an agent turns it on (`hallucination_check.confirmer`)
"""

import asyncio
import threading
from typing import Any, NamedTuple, Protocol

from agent_platform.config import get_settings
from agent_platform.config.settings import ClaimConfirmerSettings
from agent_platform.llm.local_laya import (
    Fetcher,
    Loader,
    checkpoint_folder,
    fetch_checkpoint,
    load_laya,
)
from agent_platform.logging import get_logger


class Confirmation(NamedTuple):
    """The model's answer: whether it confirms, how sure it is, and the option it chose."""

    confirmed: bool
    score: float
    label: str


class ClaimConfirmer(Protocol):
    """Reads a claim beside its source's text; says whether the source supports it."""

    @property
    def model_name(self) -> str: ...

    async def confirm(self, claim: str, evidence: str) -> Confirmation: ...


class LayaClaimConfirmer:
    """A Laya decision checkpoint asked one choice question per claim."""

    def __init__(
        self,
        settings: ClaimConfirmerSettings,
        loader: Loader | None = None,
        fetcher: Fetcher | None = None,
    ) -> None:
        self._settings = settings
        self._question = {
            settings.question_id: {
                "type": "choice",
                "instructions": settings.instructions,
                "criteria": dict(settings.criteria),
            }
        }
        self._loader = loader or load_laya
        self._fetcher = fetcher or fetch_checkpoint
        self._agent: Any = None
        self._lock = threading.Lock()
        self._logger = get_logger("hallucination_check")

    @property
    def model_name(self) -> str:
        return f"{self._settings.model}/{self._settings.subfolder}"

    def ensure_loaded_sync(self) -> None:
        """Load now, in this thread — before an event loop owns the process (macOS)."""
        with self._lock:
            if self._agent is None:
                folder = checkpoint_folder(
                    self._settings.resolved_local_model_dir,
                    self._settings.model,
                    self._settings.subfolder,
                    download=self._settings.download_if_missing,
                    fetcher=self._fetcher,
                )
                self._agent = self._loader(folder, self._settings.device)
                self._logger.info("Claim confirmer loaded", model=self.model_name)

    async def confirm(self, claim: str, evidence: str) -> Confirmation:
        """Supported, at or above `min_score`, or not confirmed (and why, as the label)."""
        await asyncio.to_thread(self.ensure_loaded_sync)
        state = {self._settings.claim_key: claim, self._settings.evidence_key: evidence}
        result = await asyncio.to_thread(self._agent.system_one, state, self._question)
        probabilities = result["answers"][self._settings.question_id]["probabilities"]
        label = max(probabilities, key=probabilities.get)
        score = float(probabilities[label])
        confirmed = label == self._settings.supported_label and score >= self._settings.min_score
        return Confirmation(confirmed=confirmed, score=score, label=label)


def create_claim_confirmer(
    settings: ClaimConfirmerSettings | None = None,
) -> ClaimConfirmer | None:
    """The configured confirmer, or None when the agent has not turned one on."""
    chosen = settings or get_settings().hallucination_check.confirmer
    return LayaClaimConfirmer(chosen) if chosen.enabled else None


def preload_claim_confirmer() -> None:
    """Load the confirmer's checkpoint now, if one is configured (before the TUI starts)."""
    confirmer = create_claim_confirmer()
    if isinstance(confirmer, LayaClaimConfirmer):
        confirmer.ensure_loaded_sync()


__all__ = [
    "ClaimConfirmer",
    "Confirmation",
    "LayaClaimConfirmer",
    "create_claim_confirmer",
    "preload_claim_confirmer",
]
