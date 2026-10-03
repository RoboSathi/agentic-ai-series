"""LLM module for the agent platform."""

from agent_platform.llm.cascade import (
    CascadingLLMClient,
    ConfiguredEscalationPolicy,
    EscalationPolicy,
)
from agent_platform.llm.client import (
    GrammarConstrainedClient,
    HostedClient,
    LLMClient,
    OpenAICompatibleClient,
    create_llm_client,
)
from agent_platform.llm.embeddings import (
    EmbeddingClient,
    HashingEmbeddingClient,
    HostedEmbeddingClient,
    cosine_similarity,
    create_embedding_client,
    tokenize,
)
from agent_platform.llm.types import (
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
