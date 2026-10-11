"""Who is believed: a chunk's trust class, and how classes rank against each other.

A retrieved copy of an order and the order row read this turn can disagree; so can what the
model said two turns ago and what a tool returned since. Which side the window keeps is a
question of provenance, answered here once (`context.trust`) for every check that has to
choose — a clash, a poisoned claim, two copies of one fact.

A source sets `metadata["trust"]` when it knows better than its kind does (rows read from the
system of record this turn are `verified`, not `retrieved`).
"""

from agent_platform.config import get_settings
from agent_platform.config.settings import TrustClass
from agent_platform.context.types import TRUST, Chunk


def trust_of(chunk: Chunk) -> TrustClass:
    """The chunk's own trust, else its source prefix's, else its kind's."""
    stated = chunk.metadata.get(TRUST)
    settings = get_settings().context.trust
    if stated in settings.rank:
        return stated  # type: ignore[no-any-return]
    for prefix, trust in settings.by_source_prefix.items():
        if chunk.source.startswith(prefix):
            return trust
    return settings.by_kind.get(chunk.kind, "user")


def rank(trust: TrustClass) -> int:
    """How far a trust class is believed; higher wins."""
    return get_settings().context.trust.rank[trust]


def chunk_rank(chunk: Chunk) -> int:
    return rank(trust_of(chunk))


def with_trust(chunk: Chunk) -> Chunk:
    """The chunk with its resolved trust written into its metadata."""
    if chunk.metadata.get(TRUST) == trust_of(chunk):
        return chunk
    return chunk.model_copy(update={"metadata": {**chunk.metadata, TRUST: trust_of(chunk)}})


__all__ = ["chunk_rank", "rank", "trust_of", "with_trust"]
