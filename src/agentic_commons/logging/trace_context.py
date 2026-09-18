"""Ambient trace id for log records.

Lives in `logging` (layer 0) rather than `observability` (layer 5) so that
every subsystem can stamp its log lines with the current trace without
importing a higher layer. `observability` is what sets it.
"""

from contextvars import ContextVar

_current_trace_id: ContextVar[str | None] = ContextVar("current_trace_id", default=None)


def get_current_trace_id() -> str | None:
    """The trace id of the request being handled on this task, if any."""
    return _current_trace_id.get()


def set_current_trace_id(trace_id: str | None) -> None:
    """Bind a trace id to the current context (observability owns calling this)."""
    _current_trace_id.set(trace_id)
