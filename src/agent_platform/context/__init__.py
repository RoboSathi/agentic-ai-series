"""Context assembly, management strategies and retrieval (LLD §7)."""

from agent_platform.context.failure import ContextFailureDetector, is_near_duplicate
from agent_platform.context.pipeline import (
    CandidateSource,
    ContextAssembler,
    FiveStagePipeline,
)
from agent_platform.context.retrieval import (
    CrossEncoderReranker,
    DenseRetriever,
    GraphRetriever,
    HybridRetriever,
    LexicalRetriever,
    RelevanceRanker,
    Retriever,
    is_identifier_query,
)
from agent_platform.context.strategies import (
    CacheStrategy,
    ConsolidateStrategy,
    ContextStrategy,
    IsolateStrategy,
    OffloadStrategy,
    ReduceStrategy,
    RetrieveStrategy,
    StrategySelector,
)
from agent_platform.context.types import (
    AssembledContext,
    Chunk,
    ChunkKind,
    Context,
    ContextFailureMode,
    ContextHealth,
    Task,
)
from agent_platform.context.window import ContextWindow, WindowResult
from agent_platform.context.working_memory import WorkingMemoryContextSource

__all__ = [
    "WindowResult",
    "AssembledContext",
    "Context",
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
    "ContextWindow",
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
    "WorkingMemoryContextSource",
    "is_identifier_query",
]
