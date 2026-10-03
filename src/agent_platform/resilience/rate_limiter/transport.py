"""httpx transports that spend the rate-limit budget per attempt.

Sitting at the transport layer is the point. Everything above it — an
SDK's own retries, `ResilientCaller` retries, a guard's status polls —
sends a real request through here, and each one is counted.

Two thin adapters over one gate, because the OpenAI SDK runs on `httpx2`
while the hosted guards and encoders use `httpx`; the two libraries share
an API but not a class hierarchy.
"""

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import httpx
import httpx2

from agent_platform.resilience.rate_limiter.limiter import RateLimiter

_TOO_MANY_REQUESTS = 429
_RETRY_AFTER = "retry-after"


class RateLimitGate:
    """The library-independent half: admit before sending, learn from a 429."""

    def __init__(self, limiter: RateLimiter) -> None:
        self._limiter = limiter

    async def admit(self, host: str) -> str | None:
        """Wait for a slot when the host is limited; return its bucket."""
        bucket = self._limiter.bucket_for_host(host)
        if bucket is not None:
            await self._limiter.acquire(bucket)
        return bucket

    def observe(self, bucket: str | None, status_code: int, retry_after: str | None) -> None:
        """Penalise the bucket when the provider says we sent too much."""
        if bucket is not None and status_code == _TOO_MANY_REQUESTS:
            self._limiter.penalize(bucket, retry_after_seconds(retry_after))


class RateLimitedTransport(httpx.AsyncBaseTransport):
    """Decorates an `httpx` transport with the rate-limit gate."""

    def __init__(self, inner: httpx.AsyncBaseTransport, limiter: RateLimiter) -> None:
        self._inner = inner
        self._gate = RateLimitGate(limiter)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        bucket = await self._gate.admit(request.url.host)
        response = await self._inner.handle_async_request(request)
        self._gate.observe(bucket, response.status_code, response.headers.get(_RETRY_AFTER))
        return response

    async def aclose(self) -> None:
        await self._inner.aclose()


class RateLimitedHttpx2Transport(httpx2.AsyncBaseTransport):
    """Decorates an `httpx2` transport (the OpenAI SDK's) with the same gate."""

    def __init__(self, inner: httpx2.AsyncBaseTransport, limiter: RateLimiter) -> None:
        self._inner = inner
        self._gate = RateLimitGate(limiter)

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        bucket = await self._gate.admit(request.url.host)
        response = await self._inner.handle_async_request(request)
        self._gate.observe(bucket, response.status_code, response.headers.get(_RETRY_AFTER))
        return response

    async def aclose(self) -> None:
        await self._inner.aclose()


def retry_after_seconds(value: str | None) -> float | None:
    """Parse a `Retry-After` value, which may be delta-seconds or an HTTP date."""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        moment = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, (moment - datetime.now(timezone.utc)).total_seconds())
