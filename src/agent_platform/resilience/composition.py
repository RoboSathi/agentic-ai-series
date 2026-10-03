"""Composing the resilience patterns (LLD §13, Decorator).

The stacking order is fixed and it is not arbitrary:

    bulkhead → circuit breaker → retry → timeout → operation

Read outermost-in. The bulkhead admits the call at all; the breaker then
decides whether this dependency is worth calling; retry re-runs the
attempt; and the timeout bounds each individual attempt, not the whole
retry budget.

Get it backwards and each layer undoes the one it should be protecting:
a timeout outside retry cuts off the retries instead of the call, and a
breaker inside retry counts one logical call as three failures and trips
at a third of the intended threshold.
"""

from typing import Awaitable, Callable, TypeVar

from agent_platform.logging import get_logger, instrument
from agent_platform.resilience.bulkhead.bulkhead import Bulkhead, ResourcePoolBulkhead
from agent_platform.resilience.circuit_breaker.breaker import CircuitBreaker, CircuitBreakerImpl
from agent_platform.resilience.retry.retry import ExponentialBackoffRetry, RetryPolicy
from agent_platform.resilience.timeout.timeout import AsyncTimeoutGuard, TimeoutGuard

T = TypeVar("T")


@instrument("resilience")
class ResilientCaller:
    """One dependency's full protection stack, applied in the fixed order.

    One instance per dependency, not one shared instance: the breaker's
    failure count and the bulkhead's permits are per-dependency state,
    and sharing them would couple the failures of unrelated services.

    Each layer is typed by its Protocol, so an agent can hand in its own
    breaker (say, a `TimeoutOnlyCircuitBreaker`) or retry policy and keep
    the stacking order, which is the part that is easy to get wrong.
    """

    def __init__(
        self,
        name: str,
        breaker: CircuitBreaker | None = None,
        retry: RetryPolicy | None = None,
        timeout: TimeoutGuard | None = None,
        bulkhead: Bulkhead | None = None,
    ) -> None:
        self.name = name
        self.breaker: CircuitBreaker = breaker or CircuitBreakerImpl(name=name)
        self.retry: RetryPolicy = retry or ExponentialBackoffRetry()
        self.timeout: TimeoutGuard = timeout or AsyncTimeoutGuard()
        self.bulkhead: Bulkhead = bulkhead or ResourcePoolBulkhead()
        self._logger = get_logger("resilience")

    async def call(
        self,
        fn: Callable[[], Awaitable[T]],
        idempotency_key: str,
        timeout_seconds: float | None = None,
        wait_seconds: float | None = None,
    ) -> T:
        """Run an operation through the whole stack."""

        async def attempt() -> T:
            return await self.timeout.run(fn, timeout_seconds)

        async def guarded() -> T:
            return await self.retry.execute(attempt, idempotency_key)

        async with self.bulkhead.acquire(self.name, wait_seconds):
            return await self.breaker.call(guarded)

    @property
    def attempts(self) -> int:
        """Total attempts made through this stack, retries included."""
        return self.retry.attempts

    @property
    def healthy(self) -> bool:
        """Whether this dependency is currently considered callable."""
        return self.breaker.state != "open"
