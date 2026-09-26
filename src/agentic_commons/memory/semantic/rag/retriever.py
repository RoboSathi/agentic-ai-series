"""Hybrid passage retrieval: sparse + dense → RRF → cross-encoder rerank.

    query ─┬─ sparse leg: BM25 over passage text   ─┐
           └─ dense leg:  KNN over passage vectors ─┴─ RRF ─ shortlist ─ rerank ─ top k

The two legs fail differently, which is why both run. BM25 finds "SKU
4471" and misses "money back" when the text says "refund"; embeddings do
the reverse. RRF fuses them by rank, so neither leg's score scale can
dominate. The cross-encoder then reads query and passage *together* —
too slow for a corpus, exactly right for a shortlist of ten.

BM25 and the reranker are the platform's own (`context.retrieval`), not
second copies: E19 teaches them once, and RAG composes them.
"""

from agentic_commons.config.settings import RagRetrievalSettings
from agentic_commons.context.retrieval import CrossEncoderReranker, LexicalRetriever
from agentic_commons.context.types import Chunk
from agentic_commons.llm.embeddings import EmbeddingClient
from agentic_commons.logging import get_logger
from agentic_commons.memory.semantic.rag.fusion import ranks, reciprocal_rank_fusion
from agentic_commons.memory.semantic.rag.protocols import VectorIndex
from agentic_commons.memory.semantic.rag.types import Hit, Passage

_PASSAGE_ID = "passage_id"


class HybridPassageRetriever:
    """Retrieves passages from one index by both legs, fused and reranked."""

    def __init__(
        self,
        index: VectorIndex,
        embedder: EmbeddingClient,
        settings: RagRetrievalSettings,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self._index = index
        self._embedder = embedder
        self._settings = settings
        self._reranker = reranker
        self._lexical = LexicalRetriever()
        self._passages: dict[str, Passage] = {}
        self._logger = get_logger("memory")
        self.refresh()

    @property
    def size(self) -> int:
        return len(self._passages)

    def refresh(self) -> None:
        """Reload the sparse leg from the index after it changed."""
        passages = self._index.passages()
        self._passages = {passage.passage_id: passage for passage in passages}
        self._lexical.index([_to_chunk(passage) for passage in passages])

    async def retrieve(self, query: str, k: int) -> list[Hit]:
        """The k best passages for a query, with every stage's evidence."""
        sparse = await self._sparse(query)
        dense = await self._dense(query)
        fused = reciprocal_rank_fusion([sparse, dense], self._settings.rrf_k)
        shortlist = sorted(fused, key=lambda pid: fused[pid], reverse=True)
        shortlist = shortlist[: self._settings.rerank_shortlist]
        reranked = await self._rerank(query, shortlist)
        sparse_ranks, dense_ranks = ranks(sparse), ranks(dense)
        order = reranked if reranked is not None else {pid: fused[pid] for pid in shortlist}
        best = sorted(order, key=lambda pid: order[pid], reverse=True)[:k]
        return [
            Hit(
                passage=self._passages[pid],
                score=order[pid],
                sparse_rank=sparse_ranks.get(pid),
                dense_rank=dense_ranks.get(pid),
                fused_score=fused[pid],
                rerank_score=reranked.get(pid) if reranked is not None else None,
            )
            for pid in best
        ]

    async def _sparse(self, query: str) -> list[str]:
        chunks = await self._lexical.retrieve(query, self._settings.sparse_k)
        return [str(chunk.metadata[_PASSAGE_ID]) for chunk in chunks]

    async def _dense(self, query: str) -> list[str]:
        """KNN ids, or none if the query vector is not from the index's space."""
        fingerprint = self._index.fingerprint()
        if fingerprint is None:
            return []
        vector = (await self._embedder.embed([query], "query"))[0]
        if self._embedder.model_id != fingerprint.model_id or len(vector) != fingerprint.dimensions:
            # Typically the hosted embedder fell back to the local one for
            # this call. Comparing across spaces returns confident nonsense,
            # so the dense leg sits this query out and BM25 carries it.
            self._logger.warning(
                "Dense leg skipped: query embedder does not match the index",
                query_model=self._embedder.model_id,
                query_dimensions=len(vector),
                index_model=fingerprint.model_id,
                index_dimensions=fingerprint.dimensions,
            )
            return []
        return [
            pid
            for pid in self._index.nearest(vector, self._settings.dense_k)
            if pid in self._passages
        ]

    async def _rerank(self, query: str, shortlist: list[str]) -> dict[str, float] | None:
        """Cross-encoder scores for the shortlist, or None when there is no reranker."""
        if self._reranker is None or not shortlist:
            return None
        chunks = [_to_chunk(self._passages[pid]) for pid in shortlist]
        rescored = await self._reranker.rerank(query, chunks, len(chunks))
        return {str(chunk.metadata[_PASSAGE_ID]): chunk.score for chunk in rescored}


def _to_chunk(passage: Passage) -> Chunk:
    """A passage in the shape the platform's retrieval primitives read."""
    return Chunk(
        content=passage.content,
        source=passage.document_path,
        token_estimate=passage.token_estimate,
        metadata={_PASSAGE_ID: passage.passage_id},
    )
