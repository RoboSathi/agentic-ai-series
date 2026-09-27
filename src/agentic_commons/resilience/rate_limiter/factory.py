"""The process's rate limiter, built once from configuration.

One instance per process, because its counters are what an eval stage
reads back to report throttling; the *budget* itself is shared across
processes by the store, not by this object.
"""

from agentic_commons.config import get_settings
from agentic_commons.resilience.rate_limiter.limiter import SlidingWindowRateLimiter
from agentic_commons.resilience.rate_limiter.store import SqliteRateWindowStore

_limiter: SlidingWindowRateLimiter | None = None


def get_rate_limiter() -> SlidingWindowRateLimiter | None:
    """The configured limiter, or None when rate limiting is switched off."""
    global _limiter
    settings = get_settings().resilience.rate_limit
    if not settings.enabled:
        return None
    if _limiter is None:
        store = SqliteRateWindowStore(settings.resolved_store_path, settings.lock_timeout_seconds)
        _limiter = SlidingWindowRateLimiter(settings, store)
    return _limiter


def reset_rate_limiter(limiter: SlidingWindowRateLimiter | None = None) -> None:
    """Replace (or drop) the process limiter — after a config reload, or in tests."""
    global _limiter
    _limiter = limiter
