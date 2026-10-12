"""httpx transports that spend the rate-limit budget per attempt.

Sitting at the transport layer is the point. Everything above it — an
SDK's own retries, `ResilientCaller` retries, a guard's status polls —
sends a real request through here, and each one is counted.

Two thin adapters over one gate, because the OpenAI SDK runs on `httpx2`
while the hosted guards and encoders use `httpx`; the two libraries share
an API but not a class hierarchy.
"""

import json
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import httpx
import httpx2

from agent_platform.resilience import admission
from agent_platform.resilience.rate_limiter.limiter import RateLimiter

_TOO_MANY_REQUESTS = 429
_RETRY_AFTER = "retry-after"


class RateLimitGate:
    """The library-independent half: admit before sending, learn from a 429."""

    def __init__(self, limiter: RateLimiter) -> None:
        self._limiter = limiter

    async def admit(self, host: str) -> str | None:
        """Wait for a slot when the host is limited; return its bucket.

        The wait is reported (`admission.queued`), so the caller's deadline and its hedge
        clock run from admission, not from the start of the queue.
        """
        bucket = self._limiter.bucket_for_host(host)
        if bucket is None:
            admission.queued(0.0)
            return None
        admission.queuing()
        waited = 0.0
        try:
            waited = await self._limiter.acquire(bucket)
        finally:
            # Resumed even when the wait gave up (quota spent, wait budget exceeded).
            admission.queued(waited)
        return bucket

    def observe(
        self, bucket: str | None, status_code: int, retry_after: str | None, model: str = ""
    ) -> None:
        """Penalise the bucket when the provider says we sent too much (naming the model)."""
        if bucket is not None and status_code == _TOO_MANY_REQUESTS:
            self._limiter.penalize(bucket, retry_after_seconds(retry_after), model=model)


class RateLimitedTransport(httpx.AsyncBaseTransport):
    """Decorates an `httpx` transport with the rate-limit gate."""

    def __init__(self, inner: httpx.AsyncBaseTransport, limiter: RateLimiter) -> None:
        self._inner = inner
        self._gate = RateLimitGate(limiter)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        bucket = await self._gate.admit(request.url.host)
        response = await self._inner.handle_async_request(request)
        self._gate.observe(
            bucket, response.status_code, response.headers.get(_RETRY_AFTER), model_of(request)
        )
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
        self._gate.observe(
            bucket, response.status_code, response.headers.get(_RETRY_AFTER), model_of(request)
        )
        return response

    async def aclose(self) -> None:
        await self._inner.aclose()


def model_of(request: httpx.Request | httpx2.Request) -> str:
    """The model a JSON request names (`model`), else its URL path — what a 429 was for."""
    try:
        body = json.loads(request.content or b"{}")
    except (ValueError, httpx.RequestNotRead, httpx2.RequestNotRead):
        body = {}
    model = body.get("model") if isinstance(body, dict) else None
    return str(model) if model else request.url.path


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
