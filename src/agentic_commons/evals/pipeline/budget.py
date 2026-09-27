"""What a stage spent against the shared provider budget."""

from dataclasses import dataclass

from agentic_commons.resilience.rate_limiter import get_rate_limiter


@dataclass(frozen=True)
class BudgetSnapshot:
    """Process-wide rate-limiter counters at one moment."""

    requests: int = 0
    throttled: int = 0
    waited_seconds: float = 0.0
    rejections: int = 0

    @classmethod
    def now(cls) -> "BudgetSnapshot":
        limiter = get_rate_limiter()
        if limiter is None:
            return cls()
        buckets = limiter.stats.buckets.values()
        return cls(
            requests=sum(bucket.requests for bucket in buckets),
            throttled=sum(bucket.throttled for bucket in buckets),
            waited_seconds=sum(bucket.waited_seconds for bucket in buckets),
            rejections=sum(bucket.rejections for bucket in buckets),
        )

    def since(self, earlier: "BudgetSnapshot") -> "BudgetSnapshot":
        return BudgetSnapshot(
            requests=self.requests - earlier.requests,
            throttled=self.throttled - earlier.throttled,
            waited_seconds=round(self.waited_seconds - earlier.waited_seconds, 2),
            rejections=self.rejections - earlier.rejections,
        )


def current_limit() -> int | None:
    """The tightest configured bucket ceiling, for the run record."""
    limiter = get_rate_limiter()
    if limiter is None:
        return None
    from agentic_commons.config import get_settings

    buckets = get_settings().resilience.rate_limit.buckets
    return min((bucket.max_requests for bucket in buckets.values()), default=None)
