"""RAG value types — the ubiquitous language of the knowledge index.

Corpus    → the directory of source documents one agent owns
Document  → one source file: identity is its path, change is its hash
Passage   → a semantically coherent chunk of a Document; the unit that
            is embedded, indexed and returned
Hit       → a Passage plus the score it earned at every stage, so a
            trace can show *why* it was retrieved, not only that it was
"""

from pydantic import BaseModel, ConfigDict, Field

# How a heading path is shown when it is folded into passage text.
HEADING_SEPARATOR = " > "


class Document(BaseModel):
    """One source file from an agent's corpus."""

    model_config = ConfigDict(frozen=True)

    path: str
    content: str
    content_hash: str


class Passage(BaseModel):
    """A chunk of a Document, carrying the headings it sits under."""

    model_config = ConfigDict(frozen=True)

    passage_id: str
    document_path: str
    ordinal: int
    heading_path: tuple[str, ...] = ()
    content: str
    token_estimate: int

    @property
    def title(self) -> str:
        """The heading path as one line, e.g. `Refunds > Timelines`."""
        return HEADING_SEPARATOR.join(self.heading_path)


class Hit(BaseModel):
    """One retrieved passage and the evidence for its rank."""

    model_config = ConfigDict(frozen=True)

    passage: Passage
    score: float
    sparse_rank: int | None = None
    dense_rank: int | None = None
    fused_score: float
    rerank_score: float | None = None


class IndexFingerprint(BaseModel):
    """What produced the vectors in an index.

    Two vectors are comparable only if the same model made them and the
    same chunking decided what they cover. Any difference means the index
    must be rebuilt, never topped up.
    """

    model_config = ConfigDict(frozen=True)

    model_id: str
    dimensions: int
    chunking_signature: str


class SyncReport(BaseModel):
    """What one sync of the corpus into the index did."""

    added: list[str] = Field(default_factory=list)
    updated: list[str] = Field(default_factory=list)
    unchanged: list[str] = Field(default_factory=list)
    removed: list[str] = Field(default_factory=list)
    passages_written: int = 0
    rebuilt: bool = False
    fingerprint: IndexFingerprint | None = None
    # Why the sync stopped early, if it did. A stopped sync leaves the
    # index consistent: documents not yet written keep their old hashes
    # and are picked up by the next sync.
    halted: str | None = None


class KnowledgeBaseStatus(BaseModel):
    """What an index currently holds, for `/kb` and the trace."""

    db_path: str
    corpus_dir: str
    documents: int
    passages: int
    fingerprint: IndexFingerprint | None = None
    last_synced: str | None = None
