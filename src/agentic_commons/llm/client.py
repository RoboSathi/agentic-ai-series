"""LLM provider adapters — the only place a provider SDK is imported (LLD §3)."""

import time
from abc import ABC, abstractmethod
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, omit
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)

from agentic_commons.config import get_settings
from agentic_commons.config.settings import LLMProviderSettings, LLMSettings, ProviderName
from agentic_commons.llm.types import (
    CompletionRequest,
    CompletionResult,
    HealthCheckResult,
    Message,
    ModelToolCall,
    TokenUsage,
)
from agentic_commons.logging import get_logger

_PLACEHOLDER_API_KEY = "not-needed"


class LLMClient(ABC):
    """Every provider adapter implements this contract."""

    @abstractmethod
    async def complete(self, request: CompletionRequest) -> CompletionResult:
        """Run one completion, returning content and/or requested tool calls."""

    @abstractmethod
    async def health_check(self) -> HealthCheckResult:
        """Check that the configured endpoint is reachable and authorized."""

    @property
    @abstractmethod
    def tier(self) -> str:
        """Name of this adapter's tier, used for cost attribution."""


def to_provider_message(message: Message) -> ChatCompletionMessageParam:
    """Convert a provider-neutral Message into the OpenAI wire format."""
    if message.role == "system":
        return ChatCompletionSystemMessageParam(role="system", content=message.content)
    if message.role == "tool":
        return ChatCompletionToolMessageParam(
            role="tool", content=message.content, tool_call_id=message.tool_call_id or ""
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
        return ChatCompletionAssistantMessageParam(role="assistant", content=message.content)
    return ChatCompletionUserMessageParam(role="user", content=message.content)


def _tool_calls_from(raw: Any) -> list[ModelToolCall] | None:
    """Extract provider-neutral tool calls from a provider response message."""
    if not raw:
        return None
    return [
        ModelToolCall(id=call.id, name=call.function.name, arguments=call.function.arguments)
        for call in raw
        if getattr(call, "type", "function") == "function"
    ]


# Payload logging truncates each message: a full 100k-token window in a
# log line is unreadable, and the first part is what identifies the bug.
_PAYLOAD_CHARS = 2000


class OpenAICompatibleClient(LLMClient):
    """Adapter for any OpenAI-compatible chat/completions endpoint.

    NVIDIA NIM and a local llama.cpp server both speak this wire format, so
    the provider is a configuration value rather than a separate code path;
    `NvidiaNimClient` and `LocalLlamaCppClient` differ only in which config
    section and API key they bind to.
    """

    provider_name: ProviderName = "nim"

    def __init__(
        self,
        settings: LLMSettings | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self._settings = settings or get_settings().llm
        self._provider: LLMProviderSettings = self._settings.provider_settings(self.provider_name)
        self._logger = get_logger("llm")
        self._client = client or AsyncOpenAI(
            base_url=self._provider.base_url,
            api_key=self._settings.api_key_for(self.provider_name) or _PLACEHOLDER_API_KEY,
            timeout=self._provider.timeout_seconds,
        )

    @property
    def tier(self) -> str:
        return self.provider_name

    @property
    def model(self) -> str:
        return self._provider.model

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        """Run one completion; max_tokens is always bounded by config."""
        started = time.perf_counter()
        self._log_request(request)
        completion = await self._client.chat.completions.create(
            model=self._provider.model,
            messages=[to_provider_message(m) for m in request.messages],
            temperature=request.temperature if request.temperature is not None else omit,
            max_tokens=request.max_tokens or self._settings.max_tokens,
            # ToolRegistry emits plain wire-format dicts so that
            # agentic_commons.tools stays provider-neutral.
            tools=request.tools if request.tools else omit,  # type: ignore[arg-type]
            **self._structured_output_kwargs(request),
        )
        choice = completion.choices[0]
        usage = self._usage_from(completion.usage)
        result = CompletionResult(
            content=choice.message.content or "",
            tool_calls=_tool_calls_from(choice.message.tool_calls),
            model=completion.model,
            finish_reason=choice.finish_reason,
            usage=usage,
            cost_usd=self.cost_for(usage),
            tier=self.tier,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        self._log_response(result)
        return result

    def _log_request(self, request: CompletionRequest) -> None:
        """Record exactly what the model was asked.

        The most useful line in the log when an agent misbehaves: what
        the model actually received is rarely what you assumed it
        received. Written to the file log only, never the curated feed.
        """
        if not self._settings.log_payloads:
            return
        self._logger.debug(
            "LLM request",
            model=self._provider.model,
            tools=[tool["function"]["name"] for tool in request.tools or []],
            messages=[
                {"role": message.role, "content": message.content[:_PAYLOAD_CHARS]}
                for message in request.messages
            ],
        )

    def _log_response(self, result: CompletionResult) -> None:
        """Record what came back, including tool calls and their arguments."""
        if not self._settings.log_payloads:
            return
        self._logger.debug(
            "LLM response",
            model=result.model,
            finish_reason=result.finish_reason,
            tokens=result.usage.total_tokens,
            latency_ms=round(result.latency_ms),
            content=result.content[:_PAYLOAD_CHARS],
            tool_calls=[
                {"name": call.name, "arguments": call.arguments} for call in result.tool_calls or []
            ],
        )

    def _structured_output_kwargs(self, request: CompletionRequest) -> dict[str, Any]:
        """Provider-specific hook for schema-constrained decoding."""
        return {}

    @staticmethod
    def _usage_from(raw: Any) -> TokenUsage:
        if raw is None:
            return TokenUsage()
        return TokenUsage(
            prompt_tokens=raw.prompt_tokens or 0,
            completion_tokens=raw.completion_tokens or 0,
            total_tokens=raw.total_tokens or 0,
        )

    def cost_for(self, usage: TokenUsage) -> float:
        """Price one completion from this provider's configured rates."""
        return (
            usage.prompt_tokens / 1000 * self._provider.input_cost_per_1k_tokens
            + usage.completion_tokens / 1000 * self._provider.output_cost_per_1k_tokens
        )

    async def health_check(self) -> HealthCheckResult:
        """Check connectivity by listing models — independent of entitlement.

        Deliberately not a completion: NIM model entitlements are
        per-account, so a specific model being un-invokable is a different
        failure from the endpoint being unreachable.
        """
        started = time.perf_counter()
        try:
            await self._client.models.list()
        except (APIConnectionError, APITimeoutError, APIStatusError) as exc:
            return HealthCheckResult(
                healthy=False, latency_ms=(time.perf_counter() - started) * 1000, error=str(exc)
            )
        return HealthCheckResult(healthy=True, latency_ms=(time.perf_counter() - started) * 1000)


class NvidiaNimClient(OpenAICompatibleClient):
    """NVIDIA NIM cloud provider."""

    provider_name: ProviderName = "nim"

    def _structured_output_kwargs(self, request: CompletionRequest) -> dict[str, Any]:
        """Ask NIM for guided JSON when a schema was requested.

        NIM accepts `nvext.guided_json`, but not every hosted model honors
        it — `guardrails` therefore still validates the result rather than
        trusting the request to have constrained anything.
        """
        if not request.json_schema:
            return {}
        return {"extra_body": {"nvext": {"guided_json": request.json_schema}}}


class LocalLlamaCppClient(OpenAICompatibleClient):
    """Local llama.cpp-style server.

    llama.cpp enforces `response_format.json_schema` with a real GBNF
    grammar, so this is the backend that can actually guarantee structurally
    valid output (see guardrails.constrained_backend).
    """

    provider_name: ProviderName = "local"

    def _structured_output_kwargs(self, request: CompletionRequest) -> dict[str, Any]:
        if not request.json_schema:
            return {}
        return {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": request.json_schema},
            }
        }


_PROVIDER_CLIENTS: dict[ProviderName, type[OpenAICompatibleClient]] = {
    "nim": NvidiaNimClient,
    "local": LocalLlamaCppClient,
}


def create_llm_client(
    settings: LLMSettings | None = None, provider: ProviderName | None = None
) -> LLMClient:
    """Build the configured LLM client (Factory).

    Returns a cascading client when `llm.cascade.enabled` is set, otherwise
    the single configured provider's adapter.
    """
    from agentic_commons.llm.cascade import CascadingLLMClient

    resolved = settings or get_settings().llm
    if provider is None and resolved.cascade.enabled:
        tiers: list[LLMClient] = [
            _PROVIDER_CLIENTS[name](resolved) for name in resolved.cascade.tiers
        ]
        return CascadingLLMClient(tiers, resolved.cascade)
    return _PROVIDER_CLIENTS[provider or resolved.provider](resolved)
