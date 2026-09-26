"""RAG contracts — the one facade an agent uses, and the parts it can swap.

An agent depends on `KnowledgeBase` and nothing else. The other Protocols
are the seams inside it: where documents come from, how they are split,
and where vectors are kept. Each has one implementation today; each is a
Protocol because each is a place a real deployment plausibly differs.
"""

from typing import Protocol, runtime_checkable

from agentic_commons.memory.semantic.rag.types import (
    Document,
    Hit,
    IndexFingerprint,
    KnowledgeBaseStatus,
    Passage,
    SyncReport,
)


@runtime_checkable
class KnowledgeBase(Protocol):
    """An agent's searchable corpus: keep it in sync, then search it."""

    async def sync(self, *, rebuild: bool = False) -> SyncReport: ...

    async def search(self, query: str, k: int | None = None) -> list[Hit]: ...

    def status(self) -> KnowledgeBaseStatus: ...


@runtime_checkable
class DocumentSource(Protocol):
    """Where a corpus's documents come from."""

    def documents(self) -> list[Document]: ...


@runtime_checkable
class Chunker(Protocol):
    """Splits a document into passages."""

    async def chunk(self, document: Document) -> list[Passage]: ...

    @property
    def signature(self) -> str:
        """Identifies this chunking configuration; a change forces a rebuild."""
        ...


@runtime_checkable
class VectorIndex(Protocol):
    """Durable storage for documents, passages and their vectors."""

    def fingerprint(self) -> IndexFingerprint | None: ...

    def reset(self, fingerprint: IndexFingerprint | None) -> None: ...

    def document_hashes(self) -> dict[str, str]: ...

    def upsert_document(
        self, document: Document, passages: list[Passage], vectors: list[list[float]]
    ) -> None: ...

    def delete_document(self, path: str) -> None: ...

    def passages(self) -> list[Passage]: ...

    def nearest(self, vector: list[float], k: int) -> list[str]: ...

    def mark_synced(self) -> None: ...

    def last_synced(self) -> str | None: ...
