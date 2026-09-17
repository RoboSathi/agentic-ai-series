"""Pub/Sub pattern stub for Phase 0."""

from typing import Any, Callable

from agentic_commons.logging import log_call


class PubSub:
    """Pub/Sub pattern stub."""

    @log_call("resilience")
    def publish(self, topic: str, event: Any) -> None:
        """Phase 0 stub — no real publishing yet."""
        pass

    @log_call("resilience")
    def subscribe(self, topic: str, handler: Callable[[Any], None]) -> None:
        """Phase 0 stub — no real subscription yet."""
        pass
