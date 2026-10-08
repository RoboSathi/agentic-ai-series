"""Every provider call, recorded as a fact (`observability.facts.LlmCallFact`).

Wraps one model's adapter, inside its resilience stack, so each attempt — a
retry, a hedged backup — is its own row, timed around the provider call and
named for the role it served and its place in that role's chain. A call that
raises is recorded too: a failure is the observation an availability SLO needs.
"""

import time

from agent_platform.llm.client import LLMClient
from agent_platform.llm.types import CompletionRequest, CompletionResult, HealthCheckResult
from agent_platform.observability.facts import FactRecorder, LlmCallFact


class RecordedClient(LLMClient):
    """Delegates every call and records what it cost and how long it took (Decorator)."""

    def __init__(self, inner: LLMClient, recorder: FactRecorder, role: str, position: int) -> None:
        self._inner = inner
        self._recorder = recorder
        self._role = role
        self._position = position

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        started = time.perf_counter()
        try:
            result = await self._inner.complete(request)
        except Exception as exc:
            self._record(self._fact(started, succeeded=False, error=repr(exc)))
            raise
        self._record(self._from_result(result, started))
        return result

    async def health_check(self) -> HealthCheckResult:
        return await self._inner.health_check()

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def tier(self) -> str:
        return self._inner.tier

    def _from_result(self, result: CompletionResult, started: float) -> LlmCallFact:
        fact = self._fact(started, succeeded=True, error="")
        return fact.model_copy(
            update={
                "model": result.model or self.model,
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "total_tokens": result.usage.total_tokens,
                "finish_reason": result.finish_reason or "",
            }
        )

    def _fact(self, started: float, *, succeeded: bool, error: str) -> LlmCallFact:
        return LlmCallFact(
            role=self._role,
            provider=self.tier,
            model=self.model,
            position=self._position,
            latency_ms=(time.perf_counter() - started) * 1000,
            succeeded=succeeded,
            error=error,
        )

    def _record(self, fact: LlmCallFact) -> None:
        self._recorder.record(fact)


__all__ = ["RecordedClient"]
