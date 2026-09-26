"""The knowledge base as a context candidate source.

Adapts `KnowledgeBase.search` to the context pipeline's `CandidateSource`
shape, so an agent adds its corpus to the window by listing one more
source — the pipeline scores, budgets and orders it like any other.
"""

from agentic_commons.context.types import Chunk, Task
from agentic_commons.memory.semantic.rag.protocols import KnowledgeBase
from agentic_commons.memory.semantic.rag.types import Hit

# Prefix on a chunk's `source`, so a trace shows it came from the corpus.
SOURCE_PREFIX = "kb:"


class KnowledgeBaseContextSource:
    """Contributes the corpus passages most relevant to the task."""

    def __init__(self, knowledge_base: KnowledgeBase, k: int | None = None) -> None:
        self._knowledge_base = knowledge_base
        self._k = k

    async def candidates(self, task: Task) -> list[Chunk]:
        if not task.query.strip():
            return []
        hits = await self._knowledge_base.search(task.query, self._k)
        return [_to_chunk(hit) for hit in hits]


def _to_chunk(hit: Hit) -> Chunk:
    passage = hit.passage
    return Chunk(
        content=passage.content,
        kind="retrieved",
        source=f"{SOURCE_PREFIX}{passage.passage_id}",
        score=hit.score,
        token_estimate=passage.token_estimate,
        metadata={
            "document": passage.document_path,
            "section": passage.title,
            "sparse_rank": hit.sparse_rank,
            "dense_rank": hit.dense_rank,
            "fused_score": hit.fused_score,
            "rerank_score": hit.rerank_score,
        },
    )
