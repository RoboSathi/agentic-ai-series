"""One model's protection: bulkhead, breaker, retry and per-attempt timeout (Decorator).

Every model call goes through the platform's `ResilientCaller`, so no agent wraps a model
in its own timeout or retry loop. One `ResilientCaller` per *model*, shared by every role on
it: the breaker's view of a model's health is the model's, not a role's.
"""

import uuid

from agent_platform.llm.client import LLMClient
from agent_platform.llm.types import CompletionRequest, CompletionResult, HealthCheckResult
from agent_platform.resilience import ResilientCaller


class ResilientLLMClient(LLMClient):
    """A model client whose every call runs through its model's resilience stack."""

    def __init__(
        self, client: LLMClient, caller: ResilientCaller, timeout_seconds: float | None = None
    ) -> None:
        self._client = client
        self._caller = caller
        self._timeout = timeout_seconds

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        """The call, retried and bounded; the result says how many attempts it took."""
        before = self._caller.attempts
        result = await self._caller.call(
            lambda: self._client.complete(request),
            idempotency_key=f"{self._caller.name}:{uuid.uuid4().hex}",
            timeout_seconds=self._timeout,
        )
        return result.model_copy(update={"attempts": max(1, self._caller.attempts - before)})

    async def health_check(self) -> HealthCheckResult:
        return await self._client.health_check()

    @property
    def model(self) -> str:
        return self._client.model

    @property
    def tier(self) -> str:
        return self._client.tier


__all__ = ["ResilientLLMClient"]
