"""LLM client interface and the OpenAI-compatible adapter that implements it."""

import time
from typing import Any, Iterator, Protocol, runtime_checkable

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, omit
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)

from agentic_commons.config.settings import LLMSettings, get_settings
from agentic_commons.llm.types import (
    HealthCheckResult,
    LLMResponse,
    Message,
    StreamedResponse,
    ToolCall,
)
from agentic_commons.logging import log_call


def _to_openai_message(message: Message) -> ChatCompletionMessageParam:
    """Convert our provider-neutral Message into the openai wire format."""
    if message.role == "system":
        return ChatCompletionSystemMessageParam(
            role="system", content=message.content
        )
    if message.role == "tool":
        return ChatCompletionToolMessageParam(
            role="tool",
            content=message.content,
            tool_call_id=message.tool_call_id or "",
        )
    if message.role == "assistant":
        if message.tool_calls:
            return ChatCompletionAssistantMessageParam(
                role="assistant",
                content=message.content or None,
                tool_calls=[
                    ChatCompletionMessageFunctionToolCallParam(
                        id=call.id,
                        type="function",
                        function={"name": call.name, "arguments": call.arguments},
                    )
                    for call in message.tool_calls
                ],
            )
        return ChatCompletionAssistantMessageParam(
            role="assistant", content=message.content
        )
    return ChatCompletionUserMessageParam(role="user", content=message.content)


def _tool_calls_from_completion(
    tool_calls: Any,
) -> list[ToolCall] | None:
    """Extract our provider-neutral ToolCall list from an openai response message."""
    if not tool_calls:
        return None
    return [
        ToolCall(id=call.id, name=call.function.name, arguments=call.function.arguments)
        for call in tool_calls
        if call.type == "function"
    ]


@runtime_checkable
class LLMClient(Protocol):
    """Interface every LLM provider adapter must implement."""

    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Generate a completion for the given conversation.

        `tools` is an OpenAI-compatible tools=[...] schema list (see
        ToolRegistry.to_openai_tools()); the response's `tool_calls` is set
        when the model wants to invoke one or more of them.
        """
        ...

    def generate_stream(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> StreamedResponse:
        """Stream a completion chunk by chunk for the given conversation."""
        ...

    def health_check(self) -> HealthCheckResult:
        """Check whether the configured LLM endpoint is reachable."""
        ...


class OpenAICompatibleClient:
    """LLMClient backed by any OpenAI-compatible chat/completions endpoint.

    NVIDIA NIM and a local llama.cpp-style server both speak this wire
    format, so one adapter driven by `LLMSettings` covers both — the
    provider is a config value, not a subclass.
    """

    def __init__(
        self,
        settings: LLMSettings | None = None,
        client: OpenAI | None = None,
    ) -> None:
        self._settings = settings or get_settings().llm
        self._client = client or OpenAI(
            base_url=self._settings.base_url,
            api_key=self._settings.api_key or "not-needed",
            timeout=self._settings.timeout_seconds,
        )

    @log_call("llm")
    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Generate a completion for the given conversation.

        max_tokens defaults to LLMSettings.max_tokens (config.yaml) when not
        given — generation is always bounded, never left to run unchecked.
        """
        completion = self._client.chat.completions.create(
            model=self._settings.model,
            messages=[_to_openai_message(m) for m in messages],
            temperature=temperature if temperature is not None else omit,
            max_tokens=max_tokens if max_tokens is not None else self._settings.max_tokens,
            # ToolRegistry.to_openai_tools() intentionally returns plain
            # dicts (openai-wire-format, not openai-SDK-typed) so
            # agentic_commons.tools stays provider-neutral.
            tools=tools if tools else omit,  # type: ignore[arg-type]
        )
        choice = completion.choices[0]
        usage = completion.usage
        return LLMResponse(
            content=choice.message.content or "",
            model=completion.model,
            tokens_used=usage.total_tokens if usage else 0,
            finish_reason=choice.finish_reason,
            tool_calls=_tool_calls_from_completion(choice.message.tool_calls),
        )

    @log_call("llm")
    def generate_stream(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> StreamedResponse:
        """Stream a completion chunk by chunk for the given conversation.

        `response` on the returned StreamedResponse is mutated in place as
        chunks are consumed and is only complete once the stream is
        exhausted — callers needing the full text must finish iterating
        first.
        """
        response = LLMResponse()

        def _chunks() -> Iterator[str]:
            stream = self._client.chat.completions.create(
                model=self._settings.model,
                messages=[_to_openai_message(m) for m in messages],
                temperature=temperature if temperature is not None else omit,
                max_tokens=max_tokens if max_tokens is not None else self._settings.max_tokens,
                stream=True,
                stream_options={"include_usage": True},
            )
            for chunk in stream:
                response.model = chunk.model
                if chunk.usage:
                    response.tokens_used = chunk.usage.total_tokens
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.finish_reason:
                    response.finish_reason = choice.finish_reason
                delta = choice.delta.content or ""
                if delta:
                    response.content += delta
                    yield delta

        return StreamedResponse(chunks=_chunks(), response=response)

    @log_call("llm")
    def health_check(self) -> HealthCheckResult:
        """Check connectivity by listing models on the configured endpoint."""
        start = time.perf_counter()
        try:
            self._client.models.list()
        except (APIConnectionError, APITimeoutError, APIStatusError) as exc:
            return HealthCheckResult(
                healthy=False,
                latency_ms=(time.perf_counter() - start) * 1000,
                error=str(exc),
            )
        return HealthCheckResult(
            healthy=True,
            latency_ms=(time.perf_counter() - start) * 1000,
        )


def create_llm_client(settings: LLMSettings | None = None) -> LLMClient:
    """Factory for the configured LLM client (Strategy/Factory pattern)."""
    return OpenAICompatibleClient(settings)
