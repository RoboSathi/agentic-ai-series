"""Memory value types (LLD §8, CoALA).

Bi-temporality is the piece most implementations collapse. `event_time`
is when the thing was true; `ingestion_time` is when we learned it. One
timestamp cannot answer "what did we believe last Tuesday?" — and that
is the question every post-incident review asks.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from agentic_commons.observability.types import utc_now

# The CoALA four. Which one a fact belongs in is a semantic decision about
# the fact's nature, never a storage convenience (E13).
MemoryType = Literal["working", "episodic", "semantic", "procedural"]

# How a fact came to be known. Weighted differently at retrieval (E16).
ProvenanceKind = Literal["verified_system_of_record", "agent_inference", "self_reported"]

# Where the write pipeline sent a candidate.
WriteRoute = Literal["auto_commit", "curator_queue", "discard", "refused"]


class Provenance(BaseModel):
    """Who said it, how they knew, and whether anyone confirmed it."""

    source: str
    kind: ProvenanceKind = "self_reported"
    confirmed: bool = False
    confirmed_at: datetime | None = None
    ingestion_time: datetime = Field(default_factory=utc_now)


class MemoryFact(BaseModel):
    """One remembered thing, in any of the four stores."""

    key: str
    value: str
    memory_type: MemoryType = "semantic"
    category: str = ""
    confidence: float = 0.5
    importance: float = 0.5
    provenance: Provenance
    event_time: datetime = Field(default_factory=utc_now)
    valid_until: datetime | None = None
    superseded_by: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_current(self) -> bool:
        """Whether this fact is still believed to hold."""
        if self.superseded_by is not None:
            return False
        return self.valid_until is None or self.valid_until > utc_now()


class MemoryQuery(BaseModel):
    """A read against a store.

    `text` drives fuzzy retrieval, `key` exact lookup; giving both lets a
    store try the cheap, deterministic path before the expensive one.
    """

    text: str = ""
    key: str = ""
    category: str = ""
    session_id: str = ""
    since: datetime | None = None
    until: datetime | None = None
    limit: int = 10
    include_superseded: bool = False


class ScoredFact(BaseModel):
    """A retrieved fact with its composite score broken out.

    The breakdown is kept rather than collapsed to one number so that
    "why was this recalled?" is answerable — the question that makes
    false recall diagnosable instead of mysterious.
    """

    fact: MemoryFact
    score: float
    relevance: float = 0.0
    recency: float = 0.0
    importance: float = 0.0
    authority: float = 0.0
    temporal_validity: float = 0.0


class WriteResult(BaseModel):
    """What the write pipeline decided, and why."""

    route: WriteRoute
    written: bool
    fact: MemoryFact | None = None
    reason: str = ""
    conflicts_with: list[str] = Field(default_factory=list)
    stage: str = ""


class MemoryConflict(BaseModel):
    """An incoming fact that disagrees with one already stored."""

    incoming: MemoryFact
    existing: MemoryFact
    resolution: Literal["supersede", "curator_queue", "reject"] = "curator_queue"
    reason: str = ""
