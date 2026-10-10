"""Reranking from a repository-local cross-encoder.

Given a query and a shortlist, which passage actually answers it? A
bi-encoder embeds each side once and compares vectors; a cross-encoder
attends over the pair, which is far better and far slower — so it runs
over the ten candidates retrieval surfaced, never over a corpus.

This used to call Hugging Face's hosted inference API. That host
(`api-inference.huggingface.co`) stopped resolving, and every turn
retried it, opened its circuit and fell back — the reranker was declared
and never ran. The same checkpoint now runs in-process, like the NLI
model: no key, no rate limit, no network once the artifact is on disk.
"""

import asyncio
from functools import cache

from agent_platform.config import get_settings
from agent_platform.config.settings import EncoderSettings
from agent_platform.llm.local_models import LocalSequenceClassifier
from agent_platform.logging import get_logger
from agent_platform.memory.store import RelevanceScorer
from agent_platform.observability import checks

# What an unavailable reranker scores every candidate: equal, so retrieval's
# own order stands. Zero would be equal too, but would read as "irrelevant".
_NEUTRAL = 0.5


class LocalCrossEncoderRanker:
    """Relevance of each (query, candidate) pair, from one resident cross-encoder.

    Satisfies the `RelevanceRanker` shape the NOLLM cascade offers, so the
    reranker can be handed either one. Degrades rather than raises: a
    model that cannot load latches to the fallback once and says so once.
    """

    def __init__(
        self,
        artifact: LocalSequenceClassifier | None = None,
        fallback: RelevanceScorer | None = None,
        settings: EncoderSettings | None = None,
    ) -> None:
        settings = settings or get_settings().llm.encoders
        self._artifact = artifact or shared_reranker_artifact()
        self._max_length = settings.reranker_max_length
        self._fallback = fallback
        self._unavailable = False
        self._logger = get_logger("llm")

    @property
    def model_name(self) -> str:
        return self._artifact.model_name

    def ensure_loaded_sync(self) -> None:
        """Load now, in this thread; a failure latches to the fallback."""
        try:
            self._artifact.load_sync()
        except Exception as exc:
            self._disable(exc)

    async def rank(self, query: str, candidates: list[str]) -> list[float]:
        """Relevance of each candidate to the query, in candidate order."""
        if not candidates:
            return []
        if self._unavailable:
            return await self._fallback_scores(query, candidates)
        try:
            tokenizer, model, torch = await self._artifact.load()
        except Exception as exc:
            self._disable(exc)
            return await self._fallback_scores(query, candidates)
        return await asyncio.to_thread(self._predict, tokenizer, model, torch, query, candidates)

    def _predict(
        self, tokenizer: object, model: object, torch: object, query: str, candidates: list[str]
    ) -> list[float]:
        """One batch: every pair together, a sigmoid over the model's single relevance logit."""
        inputs = tokenizer(  # type: ignore[operator]
            [query] * len(candidates),
            candidates,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_length,
        )
        with torch.no_grad():  # type: ignore[attr-defined]
            logits = model(**inputs).logits  # type: ignore[operator]
            scores = torch.sigmoid(logits[:, 0]).tolist()  # type: ignore[attr-defined]
        return [float(score) for score in scores]

    async def _fallback_scores(self, query: str, candidates: list[str]) -> list[float]:
        """What ranking looks like with no cross-encoder: the fallback, or no reordering."""
        if self._fallback is not None:
            ranked: list[float] = await self._fallback.rank(query, candidates)
            return ranked
        return [_NEUTRAL] * len(candidates)

    def _disable(self, exc: Exception) -> None:
        self._unavailable = True
        self._logger.warning(
            "Local reranker unavailable; every later shortlist uses the fallback",
            model=self._artifact.model_name,
            path=str(self._artifact.local_dir),
            error=str(exc) or type(exc).__name__,
        )


@cache
def shared_reranker_artifact() -> LocalSequenceClassifier:
    """One resident checkpoint per process, whatever `/harness` rebuilds.

    The weights are shared; each ranker keeps its own fallback, so two
    agents in one process never swap each other's.
    """
    settings = get_settings().llm.encoders
    return LocalSequenceClassifier(
        model_name=settings.reranker_model,
        local_dir=settings.resolved_reranker_local_model_dir,
        download_if_missing=settings.download_if_missing,
        purpose="reranker",
    )


@checks.recorded_dependency(checks.RERANK, "rank", lambda _ranker: checks.LOCAL_MODEL)
def create_reranker(fallback: RelevanceScorer | None = None) -> LocalCrossEncoderRanker | None:
    """The cross-encoder reranker, or None when encoders are switched off (Factory).

    None rather than a stub: a reranker that cannot rerank should be
    absent, and the pipeline then keeps retrieval's own ordering.
    """
    if not get_settings().llm.encoders.enabled:
        return None
    return LocalCrossEncoderRanker(fallback=fallback)
