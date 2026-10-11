"""The provider boundary: personal data leaves as tokens and comes back as values.

Redacting the person's message was never the whole of it: a tool result
(an order's delivery address, the customer's name in working memory)
went to the model provider verbatim. This decorator sits around an LLM
client and tokenises every message on the way out — system, user,
assistant, tool, and tool-call arguments — with the same keyed tokens the
rest of the harness uses (`[PERSON:3f9a1c2b7e10]`). What the model writes
back has those tokens put back, from this request's own map, before any
other code reads it.

So everything inside the harness — grounding, the guardrails, the tools,
the screen — works on real values, and only the provider sees tokens. A
token the model invents, or one from somewhere else, is not in the map
and stays a token: nothing is restored that this request did not send.
"""

from collections import OrderedDict
from typing import Any

from agent_platform.config import FeatureSet, get_settings
from agent_platform.llm.client import LLMClient
from agent_platform.llm.types import (
    CompletionRequest,
    CompletionResult,
    HealthCheckResult,
    Message,
    ModelToolCall,
)
from agent_platform.security.pii.detector import PIIFilterImpl


class PiiBoundary(LLMClient):
    """An LLM client whose provider never receives a detected value."""

    def __init__(self, inner: LLMClient, pii: PIIFilterImpl) -> None:
        self._inner = inner
        self._pii = pii
        # Texts already tokenised (a system prompt is sent on every call): text → result.
        self._cache: OrderedDict[str, tuple[str, dict[str, str]]] = OrderedDict()
        self._cache_size = get_settings().security.pii.boundary_cache_entries

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        tokens: dict[str, str] = {}
        outbound = request.model_copy(
            update={"messages": [self._outbound(m, tokens) for m in request.messages]}
        )
        result = await self._inner.complete(outbound)
        if not tokens:
            return result
        return result.model_copy(
            update={
                "content": _restore(result.content, tokens),
                "reasoning": _restore(result.reasoning, tokens),
                "tool_calls": [
                    call.model_copy(update={"arguments": _restore(call.arguments, tokens)})
                    for call in result.tool_calls
                ]
                if result.tool_calls
                else result.tool_calls,
            }
        )

    async def health_check(self) -> HealthCheckResult:
        return await self._inner.health_check()

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def tier(self) -> str:
        return self._inner.tier

    def __getattr__(self, name: str) -> Any:
        """Anything else a caller reads from a client is the wrapped client's."""
        return getattr(self._inner, name)

    def _outbound(self, message: Message, tokens: dict[str, str]) -> Message:
        calls = message.tool_calls
        return message.model_copy(
            update={
                "content": self._tokenize(message.content, tokens),
                "tool_calls": [self._call(call, tokens) for call in calls] if calls else calls,
            }
        )

    def _call(self, call: ModelToolCall, tokens: dict[str, str]) -> ModelToolCall:
        return call.model_copy(update={"arguments": self._tokenize(call.arguments, tokens)})

    def _tokenize(self, text: str, tokens: dict[str, str]) -> str:
        if not text:
            return text
        cached = self._cache.get(text)
        if cached is None:
            cached = self._pii.tokenize(text)
            self._cache[text] = cached
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(text)
        tokenized, found = cached
        tokens.update(found)
        return tokenized


def _restore(text: str, tokens: dict[str, str]) -> str:
    """Put back every token this request sent; anything else stays as it is."""
    for token, value in tokens.items():
        text = text.replace(token, value)
    return text


class BoundedRoles:
    """Per-role clients, each behind the provider boundary; anything else is the router's."""

    def __init__(self, roles: Any, pii: PIIFilterImpl) -> None:
        self._roles = roles
        self._pii = pii
        self._clients: dict[str, LLMClient] = {}

    def client_for(self, role: str) -> LLMClient:
        if role not in self._clients:
            self._clients[role] = PiiBoundary(self._roles.client_for(role), self._pii)
        return self._clients[role]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._roles, name)


def bounded(client: LLMClient, posture: FeatureSet, pii: PIIFilterImpl | None = None) -> LLMClient:
    """The client behind the provider boundary when the agent declares `pii`; else as it is."""
    return PiiBoundary(client, pii or PIIFilterImpl()) if posture.is_on("pii") else client


def bounded_roles(roles: Any, posture: FeatureSet, pii: PIIFilterImpl | None = None) -> Any:
    """Role clients behind the provider boundary when the agent declares `pii`."""
    return BoundedRoles(roles, pii or PIIFilterImpl()) if posture.is_on("pii") else roles
