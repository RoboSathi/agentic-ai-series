"""Ingestion — keep the index in step with the corpus, cheaply and safely.

Cheaply: a document whose content hash is unchanged is skipped, so a warm
start makes no embedding calls at all.

Safely: every vector in an index must come from one model under one
chunking configuration (the index fingerprint). A chunking change forces
a rebuild. A different embedder — including a hosted one that quietly
fell back to the local one mid-sync — halts the sync rather than mixing
spaces; the documents not yet written keep their old hashes and are
picked up next time.
"""

from agent_platform.llm.embeddings import EmbeddingClient
from agent_platform.logging import get_logger
from agent_platform.memory.semantic.rag.protocols import Chunker, DocumentSource, VectorIndex
from agent_platform.memory.semantic.rag.types import (
    Document,
    IndexFingerprint,
    Passage,
    SyncReport,
)


class IngestionPipeline:
    """Source → diff → chunk → embed → index, one document at a time."""

    def __init__(
        self,
        source: DocumentSource,
        chunker: Chunker,
        index: VectorIndex,
        embedder: EmbeddingClient,
        batch_size: int,
    ) -> None:
        self._source = source
        self._chunker = chunker
        self._index = index
        self._embedder = embedder
        self._batch_size = batch_size
        self._logger = get_logger("memory")

    async def sync(self, *, rebuild: bool = False) -> SyncReport:
        """Bring the index up to date with the corpus."""
        documents = self._source.documents()
        current = self._index.fingerprint()
        if current is None or current.chunking_signature != self._chunker.signature:
            rebuild = True
        if rebuild:
            self._index.reset(None)
        stored = self._index.document_hashes()
        report = SyncReport(rebuilt=rebuild, fingerprint=None if rebuild else current)
        report.removed = self._remove_vanished(stored, documents)
        for document in documents:
            if stored.get(document.path) == document.content_hash:
                report.unchanged.append(document.path)
                continue
            halted = await self._ingest(document, report, is_new=document.path not in stored)
            if halted:
                report.halted = halted
                break
        self._index.mark_synced()
        self._logger.info("Knowledge index synced", **report.model_dump(mode="json"))
        return report

    async def _ingest(self, document: Document, report: SyncReport, *, is_new: bool) -> str | None:
        """Write one document; return why the sync must stop, if it must."""
        passages = await self._chunker.chunk(document)
        vectors = await self._embed([passage.content for passage in passages])
        if not vectors:
            self._index.delete_document(document.path)
            return None
        fingerprint = IndexFingerprint(
            model_id=self._embedder.model_id,
            dimensions=len(vectors[0]),
            chunking_signature=self._chunker.signature,
        )
        if report.fingerprint is None:
            self._index.reset(fingerprint)
            report.fingerprint = fingerprint
        elif fingerprint != report.fingerprint:
            return self._mismatch(report.fingerprint, fingerprint)
        self._write(document, passages, vectors, report, is_new=is_new)
        return None

    def _write(
        self,
        document: Document,
        passages: list[Passage],
        vectors: list[list[float]],
        report: SyncReport,
        *,
        is_new: bool,
    ) -> None:
        self._index.upsert_document(document, passages, vectors)
        (report.added if is_new else report.updated).append(document.path)
        report.passages_written += len(passages)
        self._logger.debug(
            "Document indexed",
            document=document.path,
            passages=[passage.model_dump(mode="json") for passage in passages],
        )

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            vectors.extend(
                await self._embedder.embed(texts[start : start + self._batch_size], "passage")
            )
        return vectors

    def _remove_vanished(self, stored: dict[str, str], documents: list[Document]) -> list[str]:
        present = {document.path for document in documents}
        vanished = [path for path in stored if path not in present]
        for path in vanished:
            self._index.delete_document(path)
        return vanished

    def _mismatch(self, expected: IndexFingerprint, actual: IndexFingerprint) -> str:
        reason = (
            f"embedder produced {actual.model_id} ({actual.dimensions}d) but the index holds "
            f"{expected.model_id} ({expected.dimensions}d); rebuild to re-index under one model"
        )
        self._logger.warning("Knowledge sync halted", reason=reason)
        return reason
