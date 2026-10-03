"""Entailment from a repository-local NLI cross-encoder.

The hosted NLI endpoint went away with the host it lived on, and every
claim check silently fell back to token coverage — a heuristic that
cannot tell "refund approved" from "refund denied" when the words are
shared. This runs the same cross-encoder in-process: no key, no rate
limit, no network on the hot path once the artifact is on disk.
"""

import asyncio
from functools import cache

from agent_platform.config import get_settings
from agent_platform.config.settings import EncoderSettings
from agent_platform.evals.entailment import (
    EntailmentModel,
    EntailmentResult,
    LexicalEntailmentModel,
    entailment_from_probabilities,
)
from agent_platform.llm.local_models import LocalSequenceClassifier
from agent_platform.logging import get_logger


class LocalNliModel:
    """A local cross-encoder that reads premise and hypothesis together.

    Degrades rather than raises, like every encoder here: a model that
    cannot load latches to the lexical fallback once, says so once, and
    the claim check still gets an answer — a weaker, uncalibrated one.
    """

    def __init__(
        self,
        settings: EncoderSettings | None = None,
        fallback: EntailmentModel | None = None,
    ) -> None:
        settings = settings or get_settings().llm.encoders
        self._artifact = LocalSequenceClassifier(
            model_name=settings.nli_model,
            local_dir=settings.resolved_nli_local_model_dir,
            download_if_missing=settings.download_if_missing,
            purpose="NLI model",
        )
        self._max_length = settings.nli_max_length
        self._fallback = fallback or LexicalEntailmentModel()
        self._unavailable = False
        self._logger = get_logger("llm")

    @property
    def model_name(self) -> str:
        """The model that is actually answering, for the claim card."""
        if self._unavailable:
            return str(getattr(self._fallback, "model_name", "lexical"))
        return self._artifact.model_name

    def ensure_loaded_sync(self) -> None:
        """Load now, in this thread; a failure latches to the fallback."""
        try:
            self._artifact.load_sync()
        except Exception as exc:
            self._disable(exc)

    async def entails(self, premise: str, hypothesis: str) -> EntailmentResult:
        """Whether the premise entails the hypothesis, per the cross-encoder."""
        if self._unavailable:
            return await self._fallback.entails(premise, hypothesis)
        try:
            tokenizer, model, torch = await self._artifact.load()
        except Exception as exc:
            self._disable(exc)
            return await self._fallback.entails(premise, hypothesis)
        probabilities = await asyncio.to_thread(
            self._predict, tokenizer, model, torch, premise, hypothesis
        )
        return entailment_from_probabilities(probabilities, self._artifact.model_name)

    def _predict(
        self, tokenizer: object, model: object, torch: object, premise: str, hypothesis: str
    ) -> dict[str, float]:
        """Label → probability for one (premise, hypothesis) pair."""
        inputs = tokenizer(  # type: ignore[operator]
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_length,
        )
        with torch.no_grad():  # type: ignore[attr-defined]
            logits = model(**inputs).logits[0]  # type: ignore[operator]
            scores = torch.softmax(logits, dim=-1).tolist()  # type: ignore[attr-defined]
        id2label = getattr(getattr(model, "config", None), "id2label", {}) or {}
        return {str(id2label.get(index, index)): float(score) for index, score in enumerate(scores)}

    def _disable(self, exc: Exception) -> None:
        self._unavailable = True
        self._logger.warning(
            "Local NLI model unavailable; every later claim uses the lexical fallback",
            model=self._artifact.model_name,
            path=str(self._artifact.local_dir),
            error=str(exc) or type(exc).__name__,
        )


@cache
def shared_local_nli_model() -> LocalNliModel:
    """One resident NLI model per process.

    Every `/harness` rebuild composes a new claim checker; each loading its
    own copy of a DeBERTa checkpoint would cost seconds and hundreds of MB
    per toggle, for weights that never change.
    """
    return LocalNliModel()
