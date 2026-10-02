"""LLM module for agentic commons."""

from agentic_commons.llm.cascade import (
    CascadingLLMClient,
    ConfiguredEscalationPolicy,
    EscalationPolicy,
)
from agentic_commons.llm.client import (
    GrammarConstrainedClient,
    HostedClient,
    LLMClient,
    OpenAICompatibleClient,
    create_llm_client,
)
from agentic_commons.llm.embeddings import (
    EmbeddingClient,
    HashingEmbeddingClient,
    HostedEmbeddingClient,
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
    "GrammarConstrainedClient",
    "HostedClient",
    "Message",
    "ModelToolCall",
    "HostedEmbeddingClient",
    "OpenAICompatibleClient",
    "TokenUsage",
    "cosine_similarity",
    "create_embedding_client",
    "create_llm_client",
    "tokenize",
]
