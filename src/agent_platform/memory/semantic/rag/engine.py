"""The knowledge base facade, and the factory an agent builds it with.

An agent says where its index lives and where its corpus is (config), and
gets back something that can `sync`, `search` and report `status`. The
chunker, index, fusion and reranker behind it are this package's concern.
"""

from agent_platform.config import get_settings
from agent_platform.config.settings import RagSettings
from agent_platform.context.retrieval import CrossEncoderReranker, RelevanceRanker
from agent_platform.llm.embeddings import EmbeddingClient, create_embedding_client
from agent_platform.logging import get_logger, instrument
from agent_platform.memory.semantic.rag.chunking import SemanticChunker
from agent_platform.memory.semantic.rag.index import SqliteVecIndex
from agent_platform.memory.semantic.rag.ingestion import IngestionPipeline
from agent_platform.memory.semantic.rag.protocols import (
    DocumentSource,
    KnowledgeBase,
    VectorIndex,
)
from agent_platform.memory.semantic.rag.retriever import HybridPassageRetriever
from agent_platform.memory.semantic.rag.sources import MarkdownDirectorySource
from agent_platform.memory.semantic.rag.types import Hit, KnowledgeBaseStatus, SyncReport


@instrument("memory")
class RagKnowledgeBase:
    """`KnowledgeBase` composed from an ingestion pipeline and a retriever."""

    def __init__(
        self,
        settings: RagSettings,
        index: VectorIndex,
        pipeline: IngestionPipeline,
        retriever: HybridPassageRetriever,
    ) -> None:
        self._settings = settings
        self._index = index
        self._pipeline = pipeline
        self._retriever = retriever
        self._logger = get_logger("memory")

    async def sync(self, *, rebuild: bool = False) -> SyncReport:
        """Bring the index in step with the corpus, then reload the sparse leg."""
        report = await self._pipeline.sync(rebuild=rebuild)
        self._retriever.refresh()
        return report

    async def search(self, query: str, k: int | None = None) -> list[Hit]:
        """The best passages for a query; nothing, without a network call, if empty."""
        if self._retriever.size == 0:
            return []
        hits = await self._retriever.retrieve(query, k or self._settings.retrieval.top_k)
        self._logger.info(
            "Knowledge search",
            query=query,
            hits=[hit.model_dump(mode="json") for hit in hits],
        )
        return hits

    def status(self) -> KnowledgeBaseStatus:
        return KnowledgeBaseStatus(
            db_path=str(self._settings.resolved_db_path),
            corpus_dir=str(self._settings.resolved_corpus_dir),
            documents=len(self._index.document_hashes()),
            passages=self._retriever.size,
            fingerprint=self._index.fingerprint(),
            last_synced=self._index.last_synced(),
        )


def create_knowledge_base(
    settings: RagSettings | None = None,
    *,
    embedder: EmbeddingClient | None = None,
    reranker: RelevanceRanker | None = None,
    source: DocumentSource | None = None,
    index: VectorIndex | None = None,
) -> KnowledgeBase:
    """Build an agent's knowledge base from its `memory.semantic.rag` config (Factory).

    `reranker` is a cross-encoder-style scorer; without one, the fused
    order stands — absent rather than present and inert, as elsewhere.
    """
    platform = get_settings()
    resolved = settings or platform.memory.semantic.rag
    embeddings = embedder or create_embedding_client()
    vector_index = index or SqliteVecIndex(resolved.resolved_db_path)
    chunker = SemanticChunker(
        embeddings, resolved.chunking, platform.context.chars_per_token, resolved.embed_batch_size
    )
    documents = source or MarkdownDirectorySource(
        resolved.resolved_corpus_dir, resolved.corpus_glob
    )
    pipeline = IngestionPipeline(
        documents, chunker, vector_index, embeddings, resolved.embed_batch_size
    )
    retriever = HybridPassageRetriever(
        vector_index,
        embeddings,
        resolved.retrieval,
        CrossEncoderReranker(reranker) if reranker is not None else None,
    )
    return RagKnowledgeBase(resolved, vector_index, pipeline, retriever)
