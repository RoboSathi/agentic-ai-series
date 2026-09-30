"""Retry with exponential backoff and mandatory idempotency (LLD §13).

The idempotency key is required, not advisory. A retry is a duplicate
request by construction, and the only thing standing between "the
response was lost" and "the customer was refunded twice" is a key the
downstream side can recognise. The policy refuses to run without one.
"""

import asyncio
import random
from typing import Awaitable, Callable, Protocol, TypeVar, runtime_checkable

from agentic_commons.config import get_settings
from agentic_commons.logging import get_logger
from agentic_commons.resilience.types import (
    CircuitBreakerOpenError,
    NonIdempotentRetryError,
    QuotaExhaustedError,
    external_api_error_fields,
)

T = TypeVar("T")


@runtime_checkable
class RetryPolicy(Protocol):
    """Re-runs a failed operation under a bounded, keyed policy."""

    # Total attempts made, retries included — how a caller tells "worked
    # first time" from "recovered".
    attempts: int

    async def execute(self, fn: Callable[[], Awaitable[T]], idempotency_key: str) -> T: ...


class ExponentialBackoffRetry:
    """Bounded retries with exponential backoff and full jitter.

    Jitter is not a refinement. Without it, every client that failed
    together retries together, and the dependency is hit by a
    synchronized wave at each backoff boundary — the retry storm that
    keeps a recovering service down.
    """

    def __init__(
        self,
        max_attempts: int | None = None,
        base_delay_seconds: float | None = None,
        max_delay_seconds: float | None = None,
    ) -> None:
        settings = get_settings().resilience.retry
        self._max_attempts = max_attempts or settings.max_attempts
        self._base_delay = (
            base_delay_seconds if base_delay_seconds is not None else settings.base_delay_seconds
        )
        self._max_delay = (
            max_delay_seconds if max_delay_seconds is not None else settings.max_delay_seconds
        )
        # How many attempts this component has made in total. Read by the
        # caller to tell a call that worked first time from one that
        # needed the retry to work at all — the difference between
        # "nothing went wrong" and "something went wrong and we
        # recovered", which is the whole recovery SLO.
        self.attempts = 0
        self._enabled = settings.enabled
        self._logger = get_logger("resilience")

    async def execute(self, fn: Callable[[], Awaitable[T]], idempotency_key: str) -> T:
        """Run the operation, retrying transient failures up to the cap."""
        if not idempotency_key:
            raise NonIdempotentRetryError(
                "Refusing to retry without an idempotency key — a retried "
                "non-idempotent call can duplicate its effect"
            )
        if not self._enabled:
            self.attempts += 1
            return await fn()
        last_error: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                self.attempts += 1
                return await fn()
            except Exception as exc:
                if not self.should_retry(exc):
                    raise
                last_error = exc
                if attempt == self._max_attempts:
                    break
                delay = self._delay_for(attempt)
                fields = external_api_error_fields(exc)
                self._logger.warning(
                    "Retrying after failure",
                    key=idempotency_key,
                    attempt=attempt,
                    of=self._max_attempts,
                    delay_s=round(delay, 3),
                    error=str(exc),
                    **(fields or {}),
                )
                await asyncio.sleep(delay)
        assert last_error is not None
        raise last_error

    def should_retry(self, error: Exception) -> bool:
        """Whether this error is worth another attempt (hook).

        An open circuit is never retried: the breaker already decided the
        dependency is down, and retrying behind it only burns the budget.
        A subclass narrows this further — say, to never retry a 4xx.
        """
        return not isinstance(error, (CircuitBreakerOpenError, QuotaExhaustedError))

    def _delay_for(self, attempt: int) -> float:
        """Full-jitter backoff: a random point in [0, capped exponential]."""
        ceiling = min(self._max_delay, self._base_delay * (2 ** (attempt - 1)))
        return random.uniform(0.0, ceiling)
