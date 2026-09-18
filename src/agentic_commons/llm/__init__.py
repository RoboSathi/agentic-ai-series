"""LLM module for agentic commons."""

from agentic_commons.llm.cascade import (
    CascadingLLMClient,
    ConfiguredEscalationPolicy,
    EscalationPolicy,
)
from agentic_commons.llm.client import (
    LLMClient,
    LocalLlamaCppClient,
    NvidiaNimClient,
    OpenAICompatibleClient,
    create_llm_client,
)
from agentic_commons.llm.embeddings import (
    EmbeddingClient,
    HashingEmbeddingClient,
    NimEmbeddingClient,
    cosine_similarity,
    create_embedding_client,
    tokenize,
)
from agentic_commons.llm.types import (
    CompletionRequest,
    CompletionResult,
    HealthCheckResult,
    Message,
    ModelToolCall,
    TokenUsage,
)

__all__ = [
    "CascadingLLMClient",
    "CompletionRequest",
    "CompletionResult",
    "ConfiguredEscalationPolicy",
    "EmbeddingClient",
    "EscalationPolicy",
    "HashingEmbeddingClient",
    "HealthCheckResult",
    "LLMClient",
    "LocalLlamaCppClient",
    "Message",
    "ModelToolCall",
    "NimEmbeddingClient",
    "NvidiaNimClient",
    "OpenAICompatibleClient",
    "TokenUsage",
    "cosine_similarity",
    "create_embedding_client",
    "create_llm_client",
    "tokenize",
]
