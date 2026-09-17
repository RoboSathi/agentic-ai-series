"""Timeout pattern stub for Phase 0."""

from typing import Any, Callable

from agentic_commons.logging import log_call


class Timeout:
    """Timeout pattern stub."""

    @log_call("resilience")
    def run(
        self, fn: Callable[..., Any], seconds: float, *args: Any, **kwargs: Any
    ) -> Any:
        """Phase 0 stub — no real timeouts yet."""
        return fn(*args, **kwargs)
