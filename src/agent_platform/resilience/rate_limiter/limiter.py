"""Sliding-window rate limiting for hosted model endpoints.

A provider's quota is spent by requests, not by logical calls: an SDK
retry, a `ResilientCaller` retry and a guard's status poll each cost one.
So the limiter is applied per HTTP attempt (see `transport.py`), and the
budget is shared across processes through the configured store.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Awaitable, Callable, Protocol, runtime_checkable

from agent_platform.config.settings import RateLimitSettings
from agent_platform.logging import get_logger
from agent_platform.resilience.rate_limiter.store import RateWindowStore
from agent_platform.resilience.types import QuotaExhaustedError, RateLimitWaitExceededError

Clock = Callable[[], float]
Sleep = Callable[[float], Awaitable[None]]
_PERIOD_FORMAT = "%Y-%m"


@dataclass
class BucketStats:
    """What one bucket cost this process — read back by eval stages."""

    requests: int = 0
    throttled: int = 0
    waited_seconds: float = 0.0
    rejections: int = 0


@dataclass
class RateLimitStats:
    """Per-bucket counters for this process."""

    buckets: dict[str, BucketStats] = field(default_factory=dict)

    def bucket(self, name: str) -> BucketStats:
        return self.buckets.setdefault(name, BucketStats())


@runtime_checkable
class RateLimiter(Protocol):
    """Admits one request to a bucket, waiting as long as the budget requires."""

    def bucket_for_host(self, host: str) -> str | None: ...

    async def acquire(self, bucket: str) -> float: ...

    def penalize(self, bucket: str, retry_after_seconds: float | None, model: str = "") -> None: ...


class SlidingWindowRateLimiter:
    """At most `max_requests` per bucket in any `window_seconds` window.

    Sliding rather than fixed: a fixed minute lets 2x the limit through
    across a boundary (35 at 0:59, 35 more at 1:00), which is precisely
    the burst a provider's own limiter rejects.
    """

    def __init__(
        self,
        settings: RateLimitSettings,
        store: RateWindowStore,
        clock: Clock = time.time,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self._settings = settings
        self._store = store
        self._clock = clock
        self._sleep = sleep
        self._hosts = {
            host.casefold(): name
            for name, bucket in settings.buckets.items()
            for host in bucket.hosts
        }
        self.stats = RateLimitStats()
        self._logger = get_logger("resilience")

    def bucket_for_host(self, host: str) -> str | None:
        """The bucket a host draws on, or None when it is not limited."""
        return self._hosts.get(host.casefold())

    def limit_for(self, bucket: str) -> int:
        """The configured request ceiling for one bucket."""
        return self._settings.buckets[bucket].max_requests

    def quota_for(self, bucket: str) -> int | None:
        """The bucket's monthly quota, or None when it has none."""
        return self._settings.buckets[bucket].monthly_quota

    def quota_used(self, bucket: str) -> int:
        """Requests spent against the current month's quota, across every process."""
        return self._store.quota_used(bucket, self._period())

    async def acquire(self, bucket: str) -> float:
        """Spend the month's quota (or fail fast), then wait for a slot in `bucket`."""
        await self._spend_quota(bucket)
        return await self._wait_for_slot(bucket)

    async def _spend_quota(self, bucket: str) -> None:
        quota = self.quota_for(bucket)
        if quota is None:
            return
        period = self._period()
        if not await asyncio.to_thread(self._store.try_spend_quota, bucket, period, quota):
            self._logger.warning("Monthly quota spent", bucket=bucket, quota=quota, period=period)
            raise QuotaExhaustedError(
                f"'{bucket}' has used its {quota} requests for {period}; "
                "it resets on the 1st of next month (UTC)"
            )

    def _period(self) -> str:
        return datetime.fromtimestamp(self._clock(), tz=UTC).strftime(_PERIOD_FORMAT)

    async def _wait_for_slot(self, bucket: str) -> float:
        waited = 0.0
        stats = self.stats.bucket(bucket)
        while True:
            wait = await asyncio.to_thread(
                self._store.try_acquire,
                bucket,
                self.limit_for(bucket),
                self._settings.window_seconds,
                self._clock(),
            )
            if wait <= 0.0:
                stats.requests += 1
                stats.waited_seconds += waited
                return waited
            self._check_budget(bucket, waited + wait)
            if waited == 0.0:
                stats.throttled += 1
                self._logger.info(
                    "Rate limit reached; waiting", bucket=bucket, wait_s=round(wait, 2)
                )
            await self._sleep(wait)
            waited += wait

    def penalize(self, bucket: str, retry_after_seconds: float | None, model: str = "") -> None:
        """Honour a provider 429: stop every process spending this bucket."""
        delay = (
            retry_after_seconds
            if retry_after_seconds is not None
            else self._settings.default_retry_after_seconds
        )
        self.stats.bucket(bucket).rejections += 1
        self._logger.warning(
            "Provider rejected a request (429)", bucket=bucket, model=model, retry_after_s=delay
        )
        self._store.penalize(bucket, self._clock() + delay)

    def _check_budget(self, bucket: str, total_wait: float) -> None:
        if total_wait > self._settings.max_wait_seconds:
            raise RateLimitWaitExceededError(
                f"Bucket '{bucket}' needs {total_wait:.0f}s of waiting, "
                f"over the {self._settings.max_wait_seconds:.0f}s budget"
            )
