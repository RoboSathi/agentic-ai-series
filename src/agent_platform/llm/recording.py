"""An LLM client decorator that records what passed through it.

An eval needs to know what one case cost and what the model was shown
(the system prompt, retrieved context and tool results are the evidence
a reply may be grounded on). Wrapping the client records both without
the agent knowing it is being watched.
"""

from agent_platform.llm.client import LLMClient
from agent_platform.llm.types import (
    CompletionRequest,
    CompletionResult,
    HealthCheckResult,
)


class RecordingLLMClient(LLMClient):
    """Delegates every call, keeping each request and result (Decorator)."""

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner
        self.requests: list[CompletionRequest] = []
        self.results: list[CompletionResult] = []

    @property
    def tier(self) -> str:
        return self._inner.tier

    @property
    def model(self) -> str:
        return self._inner.model

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        self.requests.append(request)
        result = await self._inner.complete(request)
        self.results.append(result)
        return result

    async def health_check(self) -> HealthCheckResult:
        return await self._inner.health_check()

    @property
    def calls(self) -> int:
        return len(self.results)

    @property
    def prompt_tokens(self) -> int:
        return sum(result.usage.prompt_tokens for result in self.results)

    @property
    def completion_tokens(self) -> int:
        return sum(result.usage.completion_tokens for result in self.results)

    @property
    def cost_usd(self) -> float:
        return sum(result.cost_usd for result in self.results)

    def grounding_messages(self, roles: frozenset[str]) -> list[str]:
        """Distinct contents of the given roles across every request, in order seen."""
        seen: dict[str, None] = {}
        for request in self.requests:
            for message in request.messages:
                if message.role in roles and message.content:
                    seen.setdefault(message.content, None)
        return list(seen)
