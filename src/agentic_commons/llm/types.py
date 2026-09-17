"""Type definitions for LLM module."""

from dataclasses import dataclass
from typing import Iterator, Literal

from pydantic import BaseModel


class ToolCall(BaseModel):
    """A request from the LLM to invoke one tool."""

    id: str = ""
    name: str = ""
    arguments: str = ""


class Message(BaseModel):
    """A single message in a chat conversation.

    `tool_calls` is set on an assistant message that wants to invoke one or
    more tools. `tool_call_id` is set on a "tool" role message answering
    one specific ToolCall.id.
    """

    role: Literal["system", "user", "assistant", "tool"] = "user"
    content: str = ""
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


class LLMResponse(BaseModel):
    """Response from LLM generation."""

    content: str = ""
    model: str = ""
    tokens_used: int = 0
    finish_reason: str | None = None
    tool_calls: list[ToolCall] | None = None


class HealthCheckResult(BaseModel):
    """Result of an LLM endpoint connectivity check."""

    healthy: bool = False
    latency_ms: float = 0.0
    error: str | None = None


@dataclass
class StreamedResponse:
    """A live token stream.

    Iterate for text deltas as they arrive; `response` is the same
    `LLMResponse` instance, mutated in place, and is only complete once the
    stream has been fully consumed.
    """

    chunks: Iterator[str]
    response: LLMResponse

    def __iter__(self) -> Iterator[str]:
        return self.chunks
