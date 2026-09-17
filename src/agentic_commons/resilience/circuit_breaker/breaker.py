"""Circuit breaker stub for Phase 0."""

from typing import Any, Callable

from agentic_commons.logging import log_call


class CircuitBreaker:
    """Circuit breaker pattern stub."""

    @log_call("resilience")
    def call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Phase 0 stub — no real circuit breaking yet."""
        return fn(*args, **kwargs)
