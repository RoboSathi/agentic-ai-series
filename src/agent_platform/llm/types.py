"""Provider-neutral LLM request/result types (LLD §3)."""

from typing import Any, Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]


class ModelToolCall(BaseModel):
    """A tool invocation the model asked for, in wire form.

    Distinct from `tools.ToolCall`, which is the governed, typed call the
    Gatekeeper evaluates — this is only what came back over the wire.
    """

    id: str = ""
    name: str = ""
    arguments: str = ""


class Message(BaseModel):
    """One message in a conversation."""

    role: Role = "user"
    content: str = ""
    tool_calls: list[ModelToolCall] | None = None
    tool_call_id: str | None = None


class TokenUsage(BaseModel):
    """Token accounting for one completion — feeds cost and TUI counters."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class CompletionRequest(BaseModel):
    """Everything a provider needs for one completion."""

    messages: list[Message]
    tools: list[dict[str, Any]] | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    json_schema: dict[str, Any] | None = Field(
        default=None,
        description="Requested output schema; honored only by backends that "
        "support grammar-constrained decoding (see guardrails).",
    )
    # Provider-specific request fields beyond the standard ones, e.g. NIM's
    # `chat_template_kwargs` to switch a reasoning model's thinking off. Set
    # per role from config; a backend that does not know a field ignores it.
    extra_body: dict[str, Any] = Field(default_factory=dict)


class TokenLogprob(BaseModel):
    """One generated token, the log-probability the model gave it, and its top alternatives."""

    token: str
    logprob: float
    top: dict[str, float] = Field(default_factory=dict)


class CompletionResult(BaseModel):
    """One completion, plus the accounting the harness needs downstream."""

    content: str = ""
    # The model's own reasoning, when the provider returns it beside the answer.
    reasoning: str = ""
    tool_calls: list[ModelToolCall] | None = None
    model: str = ""
    finish_reason: str | None = None
    usage: TokenUsage = TokenUsage()
    cost_usd: float = 0.0
    tier: str = ""
    latency_ms: float = 0.0
    # Attempts the model's resilience stack made for this answer (1: it worked first time).
    attempts: int = 1
    # Per-token confidence of `content`, when the provider returned it (None otherwise).
    logprobs: list[TokenLogprob] | None = None


class HealthCheckResult(BaseModel):
    """Result of an endpoint connectivity check."""

    healthy: bool = False
    latency_ms: float = 0.0
    error: str | None = None
