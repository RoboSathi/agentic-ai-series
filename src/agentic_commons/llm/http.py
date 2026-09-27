"""HTTP clients for hosted model endpoints, rate-limited by construction.

Every adapter that talks to a hosted model builds its client here, so the
shared request budget is enforced in one place rather than remembered at
each call site. Hosts outside every configured bucket (a local llama.cpp
server, say) pass straight through.
"""

import httpx
import httpx2

from agentic_commons.resilience.rate_limiter import (
    RateLimitedHttpx2Transport,
    RateLimitedTransport,
    get_rate_limiter,
)


def create_http_client(timeout: float) -> httpx.AsyncClient:
    """An `httpx` client whose every request draws on the shared budget."""
    inner = httpx.AsyncHTTPTransport()
    limiter = get_rate_limiter()
    transport = inner if limiter is None else RateLimitedTransport(inner, limiter)
    return httpx.AsyncClient(timeout=timeout, transport=transport)


def create_sdk_http_client(timeout: float) -> httpx2.AsyncClient:
    """The same, as the `httpx2` client the OpenAI SDK requires."""
    inner = httpx2.AsyncHTTPTransport()
    limiter = get_rate_limiter()
    transport = inner if limiter is None else RateLimitedHttpx2Transport(inner, limiter)
    return httpx2.AsyncClient(timeout=timeout, transport=transport)
