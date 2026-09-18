"""Context assembly, management strategies and retrieval (LLD §7)."""

from agentic_commons.context.failure import ContextFailureDetector
from agentic_commons.context.pipeline import (
    CandidateSource,
    ContextAssembler,
    FiveStagePipeline,
)
from agentic_commons.context.retrieval import (
    CrossEncoderReranker,
    DenseRetriever,
    GraphRetriever,
    HybridRetriever,
    LexicalRetriever,
    RelevanceRanker,
    Retriever,
    is_identifier_query,
)
from agentic_commons.context.strategies import (
    CacheStrategy,
    ConsolidateStrategy,
    ContextStrategy,
    IsolateStrategy,
    OffloadStrategy,
    ReduceStrategy,
    RetrieveStrategy,
    StrategySelector,
)
from agentic_commons.context.types import (
    AssembledContext,
    Chunk,
    ChunkKind,
    ContextFailureMode,
    ContextHealth,
    Task,
)

__all__ = [
    "AssembledContext",
    "CacheStrategy",
    "CandidateSource",
    "Chunk",
    "ChunkKind",
    "ConsolidateStrategy",
    "ContextAssembler",
    "ContextFailureDetector",
    "is_near_duplicate",
    "ContextFailureMode",
    "ContextHealth",
    "ContextStrategy",
    "CrossEncoderReranker",
    "DenseRetriever",
    "FiveStagePipeline",
    "GraphRetriever",
    "HybridRetriever",
    "IsolateStrategy",
    "LexicalRetriever",
    "OffloadStrategy",
    "ReduceStrategy",
    "RelevanceRanker",
    "RetrieveStrategy",
    "Retriever",
    "StrategySelector",
    "Task",
    "is_identifier_query",
]
