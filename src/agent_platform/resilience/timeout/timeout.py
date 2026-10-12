"""Timeouts (LLD §13).

Nygard's point, unchanged: the dangerous failure is not the call that
errors, it is the call that never returns. An unbounded wait converts a
slow dependency into an exhausted thread pool, and then into an outage
of everything sharing it.
"""

import asyncio
from typing import Awaitable, Callable, Protocol, TypeVar, runtime_checkable

from agent_platform.config import get_settings
from agent_platform.logging import get_logger
from agent_platform.resilience import admission

T = TypeVar("T")


@runtime_checkable
class TimeoutGuard(Protocol):
    """Bounds how long an operation may take."""

    async def run(self, fn: Callable[[], Awaitable[T]], seconds: float | None = None) -> T: ...


class AsyncTimeoutGuard:
    """Wraps an operation in a hard deadline."""

    def __init__(self, default_seconds: float | None = None) -> None:
        settings = get_settings().resilience.timeout
        self._default = default_seconds if default_seconds is not None else settings.default_seconds
        self._enabled = settings.enabled
        self._logger = get_logger("resilience")

    @property
    def default_seconds(self) -> float:
        return self._default

    async def run(self, fn: Callable[[], Awaitable[T]], seconds: float | None = None) -> T:
        """Run the operation, raising TimeoutError past the deadline.

        The deadline covers the work, not the rate-limit queue: a wait for a slot pushes it
        back by as long as the wait (`resilience.admission`), so a throttled call is not
        timed out, retried and queued again.
        """
        if not self._enabled:
            return await fn()
        budget = seconds if seconds is not None else self._default
        try:
            async with asyncio.timeout(budget) as scope:
                with admission.deadline_scope(scope):
                    return await fn()
        except TimeoutError:
            self._logger.warning("Operation exceeded its deadline", budget_s=budget)
            raise
