"""Hosted encoder models — NLI and cross-encoder reranking (LLD §3).

Two jobs the harness needs an encoder for, and neither is generation:

- **Entailment.** Does this evidence *entail* this claim? Cosine
  similarity cannot answer it: "the refund was approved" and "the refund
  was denied" are near-identical strings and opposite facts. A
  cross-encoder trained on NLI reads the pair together and answers with
  a label.
- **Reranking.** Given a query and a shortlist, which passage actually
  answers it? A bi-encoder embeds each side once and compares vectors; a
  cross-encoder attends over the pair. The second is far better and far
  slower, which is why it runs over ten candidates and never over a
  corpus.

Both are BERT-family encoders served by Hugging Face's inference API,
reached with `hf_api_key` from the environment — never from config, like
every other credential here.

Every client degrades rather than raises. An entitlement problem, a cold
model, or no key at all leaves the caller with the deterministic
fallback it would have used anyway: a support agent must not stop
answering because a reranker is warming up.
"""

import asyncio
import hashlib
import json
from typing import Any, Protocol, runtime_checkable

import httpx
from pydantic import BaseModel

from agentic_commons.config import get_settings
from agentic_commons.config.settings import EncoderSettings
from agentic_commons.evals.entailment import (
    EntailmentModel,
    EntailmentResult,
    LexicalEntailmentModel,
    entailment_from_probabilities,
)
from agentic_commons.logging import get_logger
from agentic_commons.memory.store import RelevanceScorer
from agentic_commons.resilience import ResilientCaller, external_api_error_fields


class LabelScore(BaseModel):
    """One label and the probability the model gave it."""

    label: str
    score: float


@runtime_checkable
class TextPairScorer(Protocol):
    """Scores a batch of (left, right) text pairs."""

    async def score_pairs(self, pairs: list[tuple[str, str]]) -> list[list[LabelScore]]: ...


class HuggingFaceEncoderClient:
    """Calls one hosted encoder on the Hugging Face inference API.

    One class for both jobs because the wire shape is the same: a pair of
    texts in, labelled scores out. Which model answers — an NLI
    checkpoint or a reranking cross-encoder — is configuration, not code.
    """

    def __init__(
        self,
        model: str,
        settings: EncoderSettings | None = None,
        client: httpx.AsyncClient | None = None,
        resilient: ResilientCaller | None = None,
    ) -> None:
        self._settings = settings or get_settings().llm.encoders
        self._model = model
        self._client = client
        self._resilient = resilient or ResilientCaller(f"huggingface-encoder:{model}")
        self._logger = get_logger("llm")
        self._unavailable = False

    @property
    def model(self) -> str:
        return self._model

    @property
    def available(self) -> bool:
        """Whether this client is worth calling.

        Latches off after a definitive failure. Retrying a model the
        account has no entitlement for — or a host that no longer
        resolves — once per claim turns a dead endpoint into a latency
        problem that looks like a bug elsewhere. `_disable` is where it
        says so.
        """
        return not self._unavailable and bool(self._settings.api_key)

    async def score_pairs(self, pairs: list[tuple[str, str]]) -> list[list[LabelScore]]:
        """Score every pair, or return nothing if the model cannot answer."""
        if not pairs or not self.available:
            return []
        try:
            results = await asyncio.gather(*(self._score_one(pair) for pair in pairs))
        except Exception as exc:
            error_fields = external_api_error_fields(exc)
            self._disable(str(exc), error_fields=error_fields)
            return []
        return [result for result in results]

    async def _score_one(self, pair: tuple[str, str]) -> list[LabelScore]:
        """One pair, as the inference API wants it."""
        payload: dict[str, Any] = {
            "inputs": {"text": pair[0], "text_pair": pair[1]},
            "options": {"wait_for_model": self._settings.wait_for_model},
        }
        response = await self._post(payload)
        return _parse_labels(response)

    async def _post(self, payload: dict[str, Any]) -> Any:
        """POST to the model endpoint through the shared resilience stack."""
        url = f"{self._settings.base_url.rstrip('/')}/{self._model}"
        headers = {"Authorization": f"Bearer {self._settings.api_key}"}
        timeout = self._settings.timeout_seconds

        async def call() -> httpx.Response:
            if self._client is not None:
                response = await self._client.post(url, json=payload, headers=headers, timeout=timeout)
            else:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response

        response = await self._resilient.call(
            call,
            idempotency_key=self._idempotency_key(payload),
            timeout_seconds=timeout,
        )
        return response.json()

    def _idempotency_key(self, payload: dict[str, Any]) -> str:
        """Stable key for retrying one hosted encoder classification."""
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:20]
        return f"hf-encoder:{self._model}:{digest}"

    def _disable(
        self,
        reason: str,
        *,
        error_fields: dict[str, Any] | None = None,
    ) -> None:
        """Stop calling this model, and say what stopped working.

        Name the endpoint, not just the model. A configured host that no
        longer answers looks exactly like a model that is merely slow,
        and the difference is the whole diagnosis: the entailment tier
        and the reranker both degrade to their deterministic fallbacks
        here, quietly, for the life of the process. Whoever reads this
        line is trying to find out why a declared capability is not
        behaving like one.
        """
        self._unavailable = True
        self._logger.warning(
            "Hosted encoder unreachable; every later call uses the deterministic fallback",
            model=self._model,
            endpoint=f"{self._settings.base_url.rstrip('/')}/{self._model}",
            error=reason,
            **(error_fields or {}),
        )


class HostedNliModel:
    """Entailment from a hosted NLI cross-encoder, with a local fallback.

    The fallback is not a formality. This model answers the question the
    lexical one only approximates, so when it cannot be reached the
    caller still gets an answer — a weaker one, from a component that
    states its own limits.
    """

    def __init__(
        self,
        client: HuggingFaceEncoderClient | None = None,
        fallback: EntailmentModel | None = None,
    ) -> None:
        settings = get_settings().llm.encoders
        self._client = client or HuggingFaceEncoderClient(settings.nli_model, settings)
        self._fallback = fallback or LexicalEntailmentModel()
        self._logger = get_logger("llm")

    @property
    def model_name(self) -> str:
        """Expose the configured hosted NLI model for observability."""
        return self._client.model if self._client.available else self._fallback.model_name

    async def entails(self, premise: str, hypothesis: str) -> EntailmentResult:
        """Whether the premise entails the hypothesis, per the model."""
        scored = await self._client.score_pairs([(premise, hypothesis)])
        if not scored or not scored[0]:
            return await self._fallback.entails(premise, hypothesis)
        return entailment_from_probabilities(
            {item.label: item.score for item in scored[0]}, self._client.model
        )


class HostedCrossEncoderRanker:
    """Relevance from a hosted cross-encoder, for reranking a shortlist.

    Implements the same `RelevanceRanker` shape the NOLLM cascade offers,
    so the reranker can be handed either one and the pipeline does not
    know which it got.
    """

    def __init__(
        self,
        client: HuggingFaceEncoderClient | None = None,
        fallback: RelevanceScorer | None = None,
    ) -> None:
        settings = get_settings().llm.encoders
        self._client = client or HuggingFaceEncoderClient(settings.reranker_model, settings)
        self._fallback = fallback
        self._logger = get_logger("llm")

    async def rank(self, query: str, candidates: list[str]) -> list[float]:
        """Relevance of each candidate to the query, in candidate order."""
        if not candidates:
            return []
        scored = await self._client.score_pairs([(query, candidate) for candidate in candidates])
        if len(scored) != len(candidates):
            return await self._fallback_scores(query, candidates)
        return [_top_score(labels) for labels in scored]

    async def _fallback_scores(self, query: str, candidates: list[str]) -> list[float]:
        """What ranking looks like with no cross-encoder available."""
        if self._fallback is not None:
            ranked: list[float] = await self._fallback.rank(query, candidates)
            return ranked
        # Neutral, not zero: an unavailable reranker must not reorder the
        # shortlist retrieval already ranked, and scoring everything zero
        # would do exactly that.
        return [0.5] * len(candidates)


def _parse_labels(payload: Any) -> list[LabelScore]:
    """Read the API's labelled scores, whichever nesting it used.

    The inference API returns `[{label, score}, …]` for one input and
    `[[{label, score}, …]]` when it decides the input was a batch. Both
    shapes are the same answer.
    """
    rows = payload[0] if payload and isinstance(payload[0], list) else payload
    if not isinstance(rows, list):
        return []
    return [
        LabelScore(label=str(row["label"]), score=float(row["score"]))
        for row in rows
        if isinstance(row, dict) and "label" in row and "score" in row
    ]


def _top_score(labels: list[LabelScore]) -> float:
    """The winning label's probability, or neutral if the model said nothing."""
    return max((item.score for item in labels), default=0.5)


def create_reranker(fallback: RelevanceScorer | None = None) -> HostedCrossEncoderRanker | None:
    """A cross-encoder reranker, when one can actually be reached (Factory).

    Returns None rather than a stub when no encoder is configured. A
    reranker that cannot rerank should be absent, not present and inert:
    absent, the pipeline keeps retrieval's own ordering, which is the
    honest result.
    """
    settings = get_settings().llm.encoders
    if not settings.usable:
        return None
    return HostedCrossEncoderRanker(fallback=fallback)
