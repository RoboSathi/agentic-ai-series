"""LLM provider adapters — the only place a provider SDK is imported (LLD §3)."""

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    BadRequestError,
    omit,
)
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)

from agent_platform.config import get_settings
from agent_platform.config.settings import (
    LLMProviderSettings,
    LLMSettings,
    ProviderAdapter,
    ProviderName,
)
from agent_platform.llm.http import create_sdk_http_client
from agent_platform.llm.metering import record_usage
from agent_platform.llm.prompt import PromptManifest
from agent_platform.llm.types import (
    CompletionRequest,
    CompletionResult,
    HealthCheckResult,
    Message,
    ModelToolCall,
    TokenLogprob,
    TokenUsage,
)
from agent_platform.llm.typography import TypographyNormaliser
from agent_platform.observability.langsmith import instrument_sdk_client

if TYPE_CHECKING:
    from agent_platform.llm.observer import ModelCallObserver
from agent_platform.logging import get_logger

_PLACEHOLDER_API_KEY = "not-needed"
# Models that rejected the logprobs parameter in this process (`provider:model`).
_NO_LOGPROBS: set[str] = set()
_LOGPROBS = "logprob"


class LLMClient(ABC):
    """Every provider adapter implements this contract."""

    @abstractmethod
    async def complete(self, request: CompletionRequest) -> CompletionResult:
        """Run one completion, returning content and/or requested tool calls."""

    @abstractmethod
    async def health_check(self) -> HealthCheckResult:
        """Check that the configured endpoint is reachable and authorized."""

    @property
    def model(self) -> str:
        """Configured model name, when the adapter exposes one."""
        return ""

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


def _manifest_summary(manifest: PromptManifest | None) -> dict[str, Any]:
    """A request's sections and their sizes, beside the messages the log already holds whole."""
    if manifest is None:
        return {}
    return {
        "estimated_prompt_tokens": manifest.total_tokens,
        "tokenizer": manifest.tokenizer,
        "prefix_signature": manifest.prefix_signature,
        "sections": [
            {"name": section.name, "tokens": section.tokens, "source": section.source}
            for section in manifest.sections
        ],
    }


class OpenAICompatibleClient(LLMClient):
    """Adapter for any OpenAI-compatible chat/completions endpoint.

    A hosted catalogue and a local llama.cpp server both speak this wire format, so
    the provider is a configuration value rather than a separate code path: which
    `llm.providers` entry a client binds to is its `provider` (the configured default when
    none is named), and the entry's `adapter` picks `HostedClient` or
    `GrammarConstrainedClient`, which differ only in how they ask for structured output.
    """

    def __init__(
        self,
        settings: LLMSettings | None = None,
        client: AsyncOpenAI | None = None,
        provider: ProviderName | None = None,
    ) -> None:
        self._settings = settings or get_settings().llm
        self.provider_name: ProviderName = provider or self._settings.provider
        self._provider: LLMProviderSettings = self._settings.provider_settings(self.provider_name)
        self._logger = get_logger("llm")
        self._typography = TypographyNormaliser(self._settings.output_normalisation)
        # Inside a turn's trace each call is reported as a model run; elsewhere inert.
        self._client = client or instrument_sdk_client(
            AsyncOpenAI(
                base_url=self._provider.base_url,
                api_key=self._settings.api_key_for(self.provider_name) or _PLACEHOLDER_API_KEY,
                timeout=self._provider.timeout_seconds,
                http_client=create_sdk_http_client(self._provider.timeout_seconds),
            )
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
        try:
            completion = await self._create(request, self._logprob_kwargs())
        except BadRequestError as exc:
            if not self._rejects_logprobs(exc):
                raise
            completion = await self._create(request, {})
        choice = completion.choices[0]
        usage = self._usage_from(completion.usage)
        content, tool_calls, normalised = self._plain_text(
            choice.message.content or "", _tool_calls_from(choice.message.tool_calls)
        )
        result = CompletionResult(
            content=content,
            reasoning=self._reasoning_of(choice.message),
            tool_calls=tool_calls,
            model=completion.model,
            finish_reason=choice.finish_reason,
            usage=usage,
            cost_usd=self.cost_for(usage),
            tier=self.tier,
            latency_ms=(time.perf_counter() - started) * 1000,
            logprobs=self._logprobs_of(choice),
        )
        self._log_response(result, normalised)
        record_usage(usage)
        return result

    async def _create(self, request: CompletionRequest, logprobs: dict[str, Any]) -> Any:
        return await self._client.chat.completions.create(
            model=self._provider.model,
            messages=[to_provider_message(m) for m in request.messages],
            temperature=request.temperature if request.temperature is not None else omit,
            max_tokens=request.max_tokens or self._settings.max_tokens,
            # ToolRegistry emits plain wire-format dicts so that
            # agent_platform.tools stays provider-neutral.
            tools=request.tools if request.tools else omit,  # type: ignore[arg-type]
            **self._structured_output_kwargs(request),
            **logprobs,
            extra_body=request.extra_body or None,
        )

    def _logprob_kwargs(self) -> dict[str, Any]:
        """Ask for token log-probabilities, unless off, or this model has refused them."""
        settings = self._settings.logprobs
        if not settings.enabled or self._model_key in _NO_LOGPROBS:
            return {}
        kwargs: dict[str, Any] = {"logprobs": True}
        if settings.top_logprobs:
            kwargs["top_logprobs"] = settings.top_logprobs
        return kwargs

    def _rejects_logprobs(self, exc: BadRequestError) -> bool:
        """A 400 about logprobs: remember the model refuses them, and say so once."""
        if _LOGPROBS not in str(exc).lower() or not self._logprob_kwargs():
            return False
        _NO_LOGPROBS.add(self._model_key)
        self._logger.warning(
            "Model does not return logprobs; asking without them from now on",
            model=self._provider.model,
            error=str(exc),
        )
        return True

    @property
    def _model_key(self) -> str:
        return f"{self.provider_name}:{self._provider.model}"

    def _logprobs_of(self, choice: Any) -> list[TokenLogprob] | None:
        """The answer's tokens with their log-probabilities (None when none came back)."""
        content = getattr(getattr(choice, "logprobs", None), "content", None)
        if not content:
            return None
        return [
            TokenLogprob(
                token=self._typography.normalise(item.token)[0],
                logprob=float(item.logprob),
                top={
                    self._typography.normalise(alt.token)[0]: float(alt.logprob)
                    for alt in (getattr(item, "top_logprobs", None) or [])
                },
            )
            for item in content
        ]

    def _reasoning_of(self, message: Any) -> str:
        """The model's reasoning, from the first configured attribute the provider filled."""
        extra = getattr(message, "model_extra", None) or {}
        for name in self._settings.reasoning_fields:
            value = getattr(message, name, None) or extra.get(name)
            if isinstance(value, str) and value.strip():
                return self._typography.normalise(value.strip())[0]
        return ""

    def _plain_text(
        self, content: str, tool_calls: list[ModelToolCall] | None
    ) -> tuple[str, list[ModelToolCall] | None, int]:
        """Content and tool arguments with the model's typography undone, and the count."""
        plain, changed = self._typography.normalise(content)
        if tool_calls is None:
            return plain, None, changed
        calls = []
        for call in tool_calls:
            arguments, count = self._typography.normalise(call.arguments)
            changed += count
            calls.append(call.model_copy(update={"arguments": arguments}))
        return plain, calls, changed

    def _clip(self, text: str) -> str:
        """Apply the configured payload limit; 0 means keep everything."""
        limit = self._settings.payload_max_chars
        return text if limit <= 0 else text[:limit]

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
                {
                    **message.model_dump(mode="json"),
                    "content": self._clip(message.content),
                }
                for message in request.messages
            ],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            json_schema=request.json_schema,
            extra_body=request.extra_body,
            **_manifest_summary(request.manifest),
        )

    def _log_response(self, result: CompletionResult, normalised: int = 0) -> None:
        """Record what came back, including tool calls and their arguments.

        `normalised` is how many of the model's characters were rewritten to
        plain text; the logged text is the rewritten text everything reads.
        """
        if not self._settings.log_payloads:
            return
        self._logger.debug(
            "LLM response",
            model=result.model,
            tier=result.tier,
            finish_reason=result.finish_reason,
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            total_tokens=result.usage.total_tokens,
            cached_prompt_tokens=result.usage.cached_prompt_tokens,
            latency_ms=round(result.latency_ms, 2),
            cost_usd=result.cost_usd,
            normalised_chars=normalised,
            content=self._clip(result.content),
            reasoning=self._clip(result.reasoning),
            # Every token's confidence, as the model gave it (the check reads these).
            logprobs=[
                {"token": item.token, "logprob": round(item.logprob, 4)}
                for item in result.logprobs or []
            ],
            tool_calls=[
                {
                    "id": call.id,
                    "name": call.name,
                    "arguments": call.arguments,
                }
                for call in result.tool_calls or []
            ],
        )

    def _structured_output_kwargs(self, request: CompletionRequest) -> dict[str, Any]:
        """Provider-specific hook for schema-constrained decoding."""
        return {}

    @staticmethod
    def _usage_from(raw: Any) -> TokenUsage:
        if raw is None:
            return TokenUsage()
        details = getattr(raw, "prompt_tokens_details", None)
        return TokenUsage(
            prompt_tokens=raw.prompt_tokens or 0,
            completion_tokens=raw.completion_tokens or 0,
            total_tokens=raw.total_tokens or 0,
            cached_prompt_tokens=getattr(details, "cached_tokens", None) or 0,
        )

    def cost_for(self, usage: TokenUsage) -> float:
        """Price one completion from this provider's rates; a cached prompt token at its own."""
        fresh = usage.prompt_tokens - usage.cached_prompt_tokens
        return (
            fresh / 1000 * self._provider.input_cost_per_1k_tokens
            + usage.cached_prompt_tokens / 1000 * self._provider.cached_input_cost_per_1k_tokens
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


class HostedClient(OpenAICompatibleClient):
    """A hosted OpenAI-compatible catalogue (`adapter: hosted`)."""

    def _structured_output_kwargs(self, request: CompletionRequest) -> dict[str, Any]:
        """Send no guided-decoding fields: hosted catalogues reject them.

        The hosted models accept the OpenAI-compatible chat completion request
        but reject guided-decoding extensions with HTTP 400. Structured output
        is therefore validated after generation by the provider-neutral
        Pydantic/response guardrails.
        """
        return {}


class GrammarConstrainedClient(OpenAICompatibleClient):
    """A local llama.cpp-style server (`adapter: grammar`).

    llama.cpp enforces `response_format.json_schema` with a real GBNF
    grammar, so this is the backend that can actually guarantee structurally
    valid output (see guardrails.constrained_backend).
    """

    def _structured_output_kwargs(self, request: CompletionRequest) -> dict[str, Any]:
        if not request.json_schema:
            return {}
        return {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": request.json_schema},
            }
        }


_ADAPTERS: dict[ProviderAdapter, type[OpenAICompatibleClient]] = {
    "hosted": HostedClient,
    "grammar": GrammarConstrainedClient,
}


def _client_for(settings: LLMSettings, provider: ProviderName) -> OpenAICompatibleClient:
    """The adapter the provider's entry names, bound to that entry."""
    adapter = _ADAPTERS[settings.provider_settings(provider).adapter]
    return adapter(settings, provider=provider)


def create_llm_client(
    settings: LLMSettings | None = None,
    provider: ProviderName | None = None,
    observer: "ModelCallObserver | None" = None,
) -> LLMClient:
    """Build the configured LLM client (Factory).

    Returns a cascading client when `llm.cascade.enabled` is set; otherwise the configured
    provider's model, protected by the platform's resilience stack and backed up by
    `llm.chain.fallbacks` (`ModelChainBuilder`) — so even a plain call has a second model.
    """
    from agent_platform.config.settings import ModelRef
    from agent_platform.llm.call_log import RecordedClient, SteppedClient
    from agent_platform.llm.cascade import CascadingLLMClient
    from agent_platform.llm.chain import ModelChainBuilder

    resolved = settings or get_settings().llm
    if provider is None and resolved.cascade.enabled:
        tiers: list[LLMClient] = [
            RecordedClient(_client_for(resolved, name), role=name, position=position)
            for position, name in enumerate(resolved.cascade.tiers)
        ]
        return SteppedClient(CascadingLLMClient(tiers, resolved.cascade))
    name = provider or resolved.provider
    primary = ModelRef(provider=name, model=resolved.provider_settings(name).model)
    builder = ModelChainBuilder(
        lambda p, m: create_pinned_llm_client(p, m, resolved), observer, resolved
    )
    return builder.build(primary, resolved.chain, label=name)


def create_pinned_llm_client(
    provider: ProviderName, model: str, settings: LLMSettings | None = None
) -> LLMClient:
    """A single-provider client pinned to one model (Factory).

    For callers whose model is deliberately not the agent's — the eval
    judge, an endpoint probe — without a second copy of the adapter.
    """
    resolved = settings or get_settings().llm
    pinned = resolved.provider_settings(provider).model_copy(update={"model": model})
    overridden = resolved.model_copy(update={"providers": {**resolved.providers, provider: pinned}})
    return _client_for(overridden, provider)
