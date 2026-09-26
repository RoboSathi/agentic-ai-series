"""Memory value types (LLD §8, CoALA).

Bi-temporality is the piece most implementations collapse. `event_time`
is when the thing was true; `ingestion_time` is when we learned it. One
timestamp cannot answer "what did we believe last Tuesday?" — and that
is the question every post-incident review asks.
"""

from datetime import datetime
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from agentic_commons.observability.types import utc_now

# The CoALA four. Which one a fact belongs in is a semantic decision about
# the fact's nature, never a storage convenience (E13).
MemoryType = Literal["working", "episodic", "semantic", "procedural"]

# How a fact came to be known. Weighted differently at retrieval (E16).
ProvenanceKind = Literal["verified_system_of_record", "agent_inference", "self_reported"]

# Where the write pipeline sent a candidate.
WriteRoute = Literal["auto_commit", "curator_queue", "discard", "refused"]

# Every operation on memory is one of these, and each is authorized.
MemoryOperation = Literal["create", "update", "delete", "erase", "approve"]

# Who is acting on memory. Provenance is derived from this, never passed in.
PrincipalRole = Literal["customer", "agent", "system_of_record", "curator", "data_protection"]

# Which tier of the read cascade produced a result.
RecallTier = Literal["exact", "lexical", "dense"]


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
    # Whose fact this is. Reads are scoped to it: a session recalls its own
    # subject's facts, never another customer's.
    subject_id: str = ""
    # Updates never overwrite: each is a new version superseding the last.
    version: int = 1
    # Reinforcement: how often, and how recently, recall has used it.
    access_count: int = 0
    last_accessed_at: datetime | None = None
    # A tombstone: deleted facts keep their history but leave recall.
    deleted_at: datetime | None = None
    deleted_by: str | None = None

    @property
    def is_current(self) -> bool:
        """Whether this fact is still believed to hold."""
        if self.superseded_by is not None or self.deleted_at is not None:
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
    # The subject whose memory may be read. Empty reads the agent's shared
    # facts only — never every subject's.
    subject_id: str = ""


class MemoryPrincipal(BaseModel):
    """Who is acting on memory, as the composition root established it.

    A customer's message arrives as a `customer` principal because the
    code handling it says so — not because the message claims anything.
    """

    actor: str
    role: PrincipalRole
    subject_id: str = ""


class MemoryCommand(BaseModel):
    """One requested operation on memory, before anything decides it."""

    principal: MemoryPrincipal
    operation: MemoryOperation
    memory_type: MemoryType = "semantic"
    # create: free text to extract a claim from, or an explicit key/value.
    text: str = ""
    key: str = ""
    value: str = ""
    category: str = ""
    reason: str = ""
    session_id: str = ""
    # Conversational capture: consider only claims a configured pattern
    # recognises. Without it every sentence becomes a candidate memory and
    # the curator queue fills with requests nobody asserted as facts.
    require_pattern: bool = False


class RecallReport(BaseModel):
    """How a recall was answered — which tier, at what cost."""

    answered_by: RecallTier | None = None
    candidates: dict[str, int] = Field(default_factory=dict)
    skipped: list[str] = Field(default_factory=list)
    embed_calls: int = 0


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
    frequency: float = 0.0
    tier: RecallTier | None = None


class WriteResult(BaseModel):
    """What the write pipeline decided, and why."""

    route: WriteRoute
    written: bool
    fact: MemoryFact | None = None
    reason: str = ""
    conflicts_with: list[str] = Field(default_factory=list)
    stage: str = ""
    operation: MemoryOperation = "create"
    # Which deterministic tier extracted the claim, e.g. `D1:my_x_is_y`.
    tier_path: list[str] = Field(default_factory=list)
    permission: str = ""
    curator_id: int | None = None


class MemoryConflict(BaseModel):
    """An incoming fact that disagrees with one already stored."""

    incoming: MemoryFact
    existing: MemoryFact
    resolution: Literal["supersede", "curator_queue", "reject"] = "curator_queue"
    reason: str = ""


@runtime_checkable
class TextRedactor(Protocol):
    """Removes personal data from text before memory persists it.

    Deliberately narrower than the security plane's full PII filter:
    memory needs one verb, and depending on the whole filter would tie
    the storage layer to the detector's detection and minimization
    surface as well. Any redactor that can rewrite a string satisfies
    this.
    """

    def redact(self, text: str) -> str: ...
