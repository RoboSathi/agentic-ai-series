"""Reciprocal Rank Fusion (Cormack, Clarke & Büttcher, 2009).

BM25 scores are unbounded and cosine distances are not; adding them
lets whichever scale is larger win for reasons unrelated to relevance.
RRF uses ranks only, so the two legs are comparable without calibration:

    score(d) = Σ over legs of 1 / (k + rank_leg(d))

A document both legs agree on outranks one only a single leg loves,
which is the property hybrid retrieval is for.
"""

from collections.abc import Sequence


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int) -> dict[str, float]:
    """Fuse ranked id lists into one score per id (higher is better)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return scores


def ranks(ranking: Sequence[str]) -> dict[str, int]:
    """Each id's 1-based position in one leg's ranking."""
    return {item: rank for rank, item in enumerate(ranking, start=1)}
