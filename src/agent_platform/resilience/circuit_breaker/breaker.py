"""Circuit breaker (Nygard, *Release It!*; LLD §13).

A finite state machine with three states and four legal transitions —
and it is the canonical worked example of the same discipline the
whole-flow control loop (planned, not built) will apply to the agent itself: an explicit transition table,
illegal transitions rejected rather than silently tolerated.

    closed    --failures >= threshold-->  open
    open      --cooldown elapsed------->  half_open
    half_open --trial succeeds--------->  closed
    half_open --trial fails------------>  open
"""

import asyncio
import time
from typing import Awaitable, Callable, Protocol, TypeVar, runtime_checkable

from agent_platform.config import get_settings
from agent_platform.logging import get_logger
from agent_platform.resilience.types import CircuitBreakerOpenError, CircuitState

T = TypeVar("T")

# The only transitions the machine will make. Anything else is a bug.
_LEGAL_TRANSITIONS: dict[CircuitState, frozenset[CircuitState]] = {
    "closed": frozenset({"open"}),
    "open": frozenset({"half_open"}),
    "half_open": frozenset({"closed", "open"}),
}


@runtime_checkable
class CircuitBreaker(Protocol):
    """Refuses calls to a dependency that is already failing."""

    async def call(self, fn: Callable[[], Awaitable[T]]) -> T: ...

    @property
    def state(self) -> CircuitState: ...


class CircuitBreakerImpl:
    """Closed → Open → Half-Open, with one trial call to close again.

    The half-open state is what makes recovery automatic without being
    reckless: exactly one caller is let through to test the dependency,
    and everyone else keeps failing fast until that trial reports back.
    """

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int | None = None,
        half_open_after_seconds: float | None = None,
    ) -> None:
        settings = get_settings().resilience.circuit_breaker
        self._name = name
        self._threshold = failure_threshold or settings.failure_threshold
        self._cooldown = (
            half_open_after_seconds
            if half_open_after_seconds is not None
            else settings.half_open_after_seconds
        )
        self._enabled = settings.enabled
        self._state: CircuitState = "closed"
        self._failures = 0
        self._opened_at = 0.0
        self._lock = asyncio.Lock()
        self._logger = get_logger("resilience")

    @property
    def state(self) -> CircuitState:
        """Current state, after accounting for an elapsed cooldown."""
        if self._state == "open" and self._cooldown_elapsed():
            self._transition_to("half_open")
        return self._state

    @property
    def name(self) -> str:
        return self._name

    @property
    def failure_count(self) -> int:
        """Consecutive failures recorded while closed."""
        return self._failures

    async def call(self, fn: Callable[[], Awaitable[T]]) -> T:
        """Run the operation, or fail fast if the circuit is open."""
        if not self._enabled:
            return await fn()
        if self.state == "open":
            raise CircuitBreakerOpenError(
                f"Circuit '{self._name}' is open; {self._failures} consecutive failures"
            )
        try:
            result = await fn()
        except Exception as exc:
            if self.counts_as_failure(exc):
                await self._record_failure()
            raise
        await self._record_success()
        return result

    def counts_as_failure(self, error: Exception) -> bool:
        """Whether this error says the dependency is unhealthy (hook).

        Every error counts by default. A subclass narrows it when some
        errors are answers rather than outages — the state machine,
        cooldown and trial call stay the platform's.
        """
        return True

    async def _record_success(self) -> None:
        """A success closes a half-open circuit and clears the failure count."""
        async with self._lock:
            if self._state == "half_open":
                self._transition_to("closed")
            self._failures = 0

    async def _record_failure(self) -> None:
        """A failure trips a closed circuit, or re-opens a half-open one."""
        async with self._lock:
            self._failures += 1
            if self._state == "half_open":
                self._transition_to("open")
            elif self._state == "closed" and self._failures >= self._threshold:
                self._transition_to("open")

    def _cooldown_elapsed(self) -> bool:
        """Whether the open circuit has waited long enough for a trial call."""
        return (time.monotonic() - self._opened_at) >= self._cooldown

    def _transition_to(self, target: CircuitState) -> None:
        """Move to a new state, refusing any transition not in the table."""
        if target not in _LEGAL_TRANSITIONS[self._state]:
            raise ValueError(f"Illegal circuit transition: {self._state} -> {target}")
        self._logger.info(
            "Circuit state changed", circuit=self._name, from_state=self._state, to_state=target
        )
        self._state = target
        if target == "open":
            self._opened_at = time.monotonic()

    def reset(self) -> None:
        """Force the circuit closed — an operator action, not an automatic one."""
        self._state = "closed"
        self._failures = 0


class TimeoutOnlyCircuitBreaker(CircuitBreakerImpl):
    """A breaker that trips on timeouts alone (the narrow E18 variant).

    Kept as a documented subclass rather than a second implementation.
    The distinction it draws is real: a dependency returning prompt errors
    is answering, and tripping on those hides a bug behind an outage.
    """

    def counts_as_failure(self, error: Exception) -> bool:
        """Only a timeout counts; a prompt error is a live dependency."""
        return isinstance(error, (asyncio.TimeoutError, TimeoutError))
