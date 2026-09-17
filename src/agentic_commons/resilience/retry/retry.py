"""Retry pattern stub for Phase 0."""

from typing import Any, Callable

from agentic_commons.logging import log_call


class Retry:
    """Retry pattern stub."""

    @log_call("resilience")
    def execute(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Phase 0 stub — no real retries yet."""
        return fn(*args, **kwargs)
