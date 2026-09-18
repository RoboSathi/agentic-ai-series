"""The shared store interface and composite retrieval scoring (LLD §8).

Park et al.'s generative-agents retrieval combined relevance, recency and
importance. Two more signals matter once memory is a security boundary:
*authority* (a self-reported claim must not outrank a system of record)
and *temporal validity* (a fact with an expiry should stop winning after
it expires). Single-signal retrieval feels subtly wrong for a reason —
pure similarity recalls a superseded fact that happens to match best.
"""

import math
from typing import Protocol, runtime_checkable

from agentic_commons.config import get_settings
from agentic_commons.memory.types import (
    MemoryFact,
    MemoryQuery,
    Provenance,
    ScoredFact,
    WriteResult,
)
from agentic_commons.observability.types import utc_now


@runtime_checkable
class MemoryStore(Protocol):
    """One CoALA store. Implemented, not merely declared — checked by mypy."""

    def read(self, query: MemoryQuery) -> list[MemoryFact]: ...

    def write(self, fact: MemoryFact, provenance: Provenance) -> WriteResult: ...


@runtime_checkable
class RelevanceScorer(Protocol):
    """Scores a query against candidate texts.

    Structurally satisfied by `evals.NollmScoringCascadeImpl.rank`, so
    memory reuses the retrieval cascade without importing layer 5.
    """

    async def rank(self, text: str, candidates: list[str]) -> list[float]: ...


class CompositeRetrievalScorer:
    """Five weighted signals, combined into one ranking (E13, extended).

    The weights are configuration because they are a product decision:
    how much a recent-but-unverified claim should outrank an older
    verified one is a judgement call someone should be able to change and
    justify without a code change.
    """

    def __init__(self) -> None:
        settings = get_settings().memory
        self._weights = settings.retrieval_weights
        self._provenance_weights = settings.provenance_weights
        self._half_life_days = settings.recency_half_life_days

    def score(self, fact: MemoryFact, relevance: float) -> ScoredFact:
        """Combine the five signals for one fact."""
        recency = self.recency(fact)
        authority = self.authority(fact)
        validity = self.temporal_validity(fact)
        weights = self._weights
        total = (
            weights.relevance * relevance
            + weights.recency * recency
            + weights.importance * fact.importance
            + weights.authority * authority
            + weights.temporal_validity * validity
        )
        return ScoredFact(
            fact=fact,
            score=total,
            relevance=relevance,
            recency=recency,
            importance=fact.importance,
            authority=authority,
            temporal_validity=validity,
        )

    def rank(
        self, facts: list[MemoryFact], relevances: list[float], limit: int
    ) -> list[ScoredFact]:
        """Score and order a candidate set, best first."""
        scored = [self.score(fact, relevance) for fact, relevance in zip(facts, relevances)]
        return sorted(scored, key=lambda item: item.score, reverse=True)[:limit]

    def recency(self, fact: MemoryFact) -> float:
        """Exponential decay on *ingestion* time, with a configured half-life.

        Ingestion, not event time: how recently we learned something is
        what should fade, otherwise a fact about last year that we learned
        this morning is scored as stale.
        """
        age_days = (utc_now() - fact.provenance.ingestion_time).total_seconds() / 86400
        return math.pow(0.5, max(0.0, age_days) / self._half_life_days)

    def authority(self, fact: MemoryFact) -> float:
        """How much weight the fact's provenance earns it."""
        base = self._provenance_weights.get(fact.provenance.kind, 0.0)
        return min(1.0, base + 0.1) if fact.provenance.confirmed else base

    def temporal_validity(self, fact: MemoryFact) -> float:
        """Zero once superseded or expired — this is what stops false recall.

        A superseded fact that still matches the query best is exactly the
        false-positive case E16 asks for: confidently recalling the wrong
        stored fact, which is a different failure from forgetting.
        """
        if fact.superseded_by is not None:
            return 0.0
        if fact.valid_until is None:
            return 1.0
        return 1.0 if fact.valid_until > utc_now() else 0.0
