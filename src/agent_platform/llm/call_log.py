"""Every model call, recorded as facts (`observability.facts.ModelCallFact`).

Two decorators, at the two ends of a model's protection stack:

- `SteppedClient`, outermost: one logical call is one **step** of the turn,
  however many attempts and backup models it takes (UL "Step").
- `RecordedClient`, innermost, around each model's adapter: each attempt — a
  retry, a hedged backup — is its own row, timed around the provider call,
  numbered within its step and marked when a backup model served it. A call
  that raises is recorded too: a failure is the observation availability needs.
"""

import time

from agent_platform.llm.client import LLMClient
from agent_platform.llm.observer import ModelCallObserver
from agent_platform.llm.types import CompletionRequest, CompletionResult, HealthCheckResult
from agent_platform.observability.facts import FactStore, ModelCallFact, fact_store
from agent_platform.observability.unit_context import (
    Generation,
    bound_step,
    next_attempt,
    remember_generation,
)


class _Delegating(LLMClient):
    """Passes the client's identity and health through to the wrapped one."""

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        return await self._inner.complete(request)

    async def health_check(self) -> HealthCheckResult:
        return await self._inner.health_check()

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def tier(self) -> str:
        return self._inner.tier


class SteppedClient(_Delegating):
    """The outermost client of a role: each call it serves is one step (Decorator).

    A completion that came with token log-probabilities is kept on the turn, so the
    hallucination check can score the reply it produced without any agent passing
    them along. The observer hears of every request first, so a request no step card
    described still gets its CONTEXT card.
    """

    def __init__(
        self, inner: LLMClient, observer: ModelCallObserver | None = None, label: str = ""
    ) -> None:
        super().__init__(inner)
        self._observer = observer
        self._label = label

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        if self._observer is not None:
            self._observer.request_starting(self._label, self.model, request)
        with bound_step():
            result = await self._inner.complete(request)
        if result.logprobs:
            remember_generation(
                Generation(
                    model=result.model or self.model,
                    content=result.content,
                    tokens=[(item.token, item.logprob) for item in result.logprobs],
                )
            )
        return result


class RecordedClient(_Delegating):
    """Delegates every call and records what it cost and how long it took (Decorator)."""

    def __init__(
        self, inner: LLMClient, role: str, position: int, store: FactStore | None = None
    ) -> None:
        super().__init__(inner)
        self._store = store
        self._role = role
        self._position = position

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        attempt = next_attempt()
        started = time.perf_counter()
        try:
            result = await self._inner.complete(request)
        except Exception as exc:
            self._append(self._fact(started, attempt, succeeded=False, error=repr(exc)))
            raise
        self._append(self._from_result(result, started, attempt))
        return result

    def _from_result(self, result: CompletionResult, started: float, attempt: int) -> ModelCallFact:
        fact = self._fact(started, attempt, succeeded=True, error="")
        return fact.model_copy(
            update={
                "model": result.model or self.model,
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "total_tokens": result.usage.total_tokens,
                "finish_reason": result.finish_reason or "",
            }
        )

    def _fact(self, started: float, attempt: int, *, succeeded: bool, error: str) -> ModelCallFact:
        return ModelCallFact(
            role=self._role,
            provider=self.tier,
            model=self.model,
            attempt=attempt,
            is_backup=self._position > 0,
            latency_ms=(time.perf_counter() - started) * 1000,
            succeeded=succeeded,
            error=error,
        )

    def _append(self, fact: ModelCallFact) -> None:
        (self._store or fact_store()).append(fact)


__all__ = ["RecordedClient", "SteppedClient"]
