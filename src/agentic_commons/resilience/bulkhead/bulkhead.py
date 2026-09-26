"""Bulkheads — per-dependency resource pools (LLD §13).

Named after a ship's compartments: a hull breach floods one compartment,
not the vessel. Here, one dependency saturating its pool must not starve
calls to an unrelated dependency — which is exactly the property E35
asserts as an exit invariant.
"""

import asyncio
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import AsyncIterator, Protocol, runtime_checkable

from agentic_commons.config import get_settings
from agentic_commons.logging import get_logger
from agentic_commons.resilience.types import BulkheadFullError


@runtime_checkable
class Bulkhead(Protocol):
    """Limits concurrent use of one named resource pool."""

    def acquire(
        self, pool: str, wait_seconds: float | None = None
    ) -> AbstractAsyncContextManager[None]: ...


class ResourcePoolBulkhead:
    """One semaphore per named pool, created on first use.

    Pools are separate objects rather than one shared counter — that
    separation *is* the isolation. A single global limit would let a
    stalled dependency consume every permit and block the rest.
    """

    def __init__(self, pool_size: int | None = None) -> None:
        settings = get_settings().resilience.bulkhead
        self._size = pool_size or settings.pool_size
        self._enabled = settings.enabled
        self._pools: dict[str, asyncio.Semaphore] = {}
        self._in_flight: dict[str, int] = {}
        self._logger = get_logger("resilience")

    @asynccontextmanager
    async def acquire(self, pool: str, wait_seconds: float | None = None) -> AsyncIterator[None]:
        """Hold a permit for one pool, releasing it on exit.

        With no wait budget the caller queues; with one, it is rejected
        rather than left waiting — shedding load is a deliberate choice
        the caller makes, not a default.
        """
        if not self._enabled:
            yield
            return
        semaphore = self._pools.setdefault(pool, asyncio.Semaphore(self._size))
        try:
            if wait_seconds is None:
                await semaphore.acquire()
            else:
                await asyncio.wait_for(semaphore.acquire(), timeout=wait_seconds)
        except asyncio.TimeoutError as exc:
            self._logger.warning("Bulkhead pool exhausted", pool=pool, size=self._size)
            raise BulkheadFullError(f"Pool '{pool}' exhausted ({self._size} permits)") from exc
        self._in_flight[pool] = self._in_flight.get(pool, 0) + 1
        try:
            yield
        finally:
            self._in_flight[pool] -= 1
            semaphore.release()

    def in_flight(self, pool: str) -> int:
        """Calls currently holding a permit in one pool."""
        return self._in_flight.get(pool, 0)

    def available(self, pool: str) -> int:
        """Permits still free in one pool."""
        return self._size - self.in_flight(pool)

    @property
    def pool_size(self) -> int:
        return self._size
