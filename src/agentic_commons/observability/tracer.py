"""Tracer stubs for Phase 0."""

from agentic_commons.logging import log_call
from agentic_commons.observability.types import SpanHandle


class Tracer:
    """Tracing interface stub."""

    @log_call("observability")
    def start_span(self, name: str) -> SpanHandle:
        """Phase 0 stub — no real tracing yet."""
        return SpanHandle(id=f"span-{name}")

    @log_call("observability")
    def end_span(self, handle: SpanHandle) -> None:
        """Phase 0 stub — no real tracing yet."""
        pass
