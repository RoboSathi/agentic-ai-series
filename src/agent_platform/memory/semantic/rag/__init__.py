"""Document RAG for semantic memory (LLD §7/§8).

An agent points this at its own index path and corpus directory
(`memory.semantic.rag` in its config) and gets hybrid retrieval — BM25 and
dense KNN fused by RRF, then cross-encoder reranked — over semantically
chunked passages, behind one interface:

    kb = create_knowledge_base(reranker=...)
    await kb.sync()                      # hash-skipped; cheap when warm
    hits = await kb.search("how long do refunds take")

Add it to the context window with `KnowledgeBaseContextSource(kb)`.
"""

from agent_platform.memory.semantic.rag.chunking import section_sentences
from agent_platform.memory.semantic.rag.context_source import (
    SOURCE_PREFIX as KNOWLEDGE_SOURCE_PREFIX,
)
from agent_platform.memory.semantic.rag.context_source import KnowledgeBaseContextSource
from agent_platform.memory.semantic.rag.engine import RagKnowledgeBase, create_knowledge_base
from agent_platform.memory.semantic.rag.protocols import (
    Chunker,
    DocumentSource,
    KnowledgeBase,
    VectorIndex,
)
from agent_platform.memory.semantic.rag.types import (
    Document,
    Hit,
    IndexFingerprint,
    KnowledgeBaseStatus,
    Passage,
    SyncReport,
)

__all__ = [
    "Chunker",
    "Document",
    "DocumentSource",
    "Hit",
    "IndexFingerprint",
    "KNOWLEDGE_SOURCE_PREFIX",
    "KnowledgeBase",
    "KnowledgeBaseContextSource",
    "KnowledgeBaseStatus",
    "Passage",
    "RagKnowledgeBase",
    "SyncReport",
    "VectorIndex",
    "create_knowledge_base",
    "section_sentences",
]
