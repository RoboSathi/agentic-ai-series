"""Ambient request identity for log records: which agent, session, channel and trace.

Lives in `logging` (layer 0) rather than `observability` (layer 5) so that
every subsystem can stamp its log lines without importing a higher layer.

One turn is one trace: the agent binds a fresh trace id when a turn starts,
and every line any subsystem logs during it carries that id, the agent, the
session and the channel it came in on. `grep <trace_id> logs/<agent>/*.log`
then returns the whole turn across every file. Bound with a token and reset
on exit, so a turn's identity ends with the turn and never leaks into the
caller's context.
"""

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token

_current_trace_id: ContextVar[str | None] = ContextVar("current_trace_id", default=None)
_current_agent: ContextVar[str | None] = ContextVar("current_agent", default=None)
_current_session: ContextVar[str | None] = ContextVar("current_session", default=None)
_current_channel: ContextVar[str | None] = ContextVar("current_channel", default=None)

# The record fields, in the order they lead every line.
_FIELDS = (
    ("agent", _current_agent),
    ("session", _current_session),
    ("channel", _current_channel),
    ("trace_id", _current_trace_id),
)


def get_current_trace_id() -> str | None:
    """The trace id of the request being handled on this task, if any."""
    return _current_trace_id.get()


def set_current_trace_id(trace_id: str | None) -> None:
    """Bind a trace id to the current context (observability owns calling this)."""
    _current_trace_id.set(trace_id)


def new_trace_id() -> str:
    """A fresh trace id: 32 hex characters, the W3C trace-context and LangSmith shape."""
    return uuid.uuid4().hex


def new_session_id(agent_id: str, separator: str, hex_chars: int) -> str:
    """A conversation id named for the agent that owns it, not the screen that opened it."""
    return f"{agent_id}{separator}{uuid.uuid4().hex[:hex_chars]}"


@contextmanager
def bound_channel(channel: str) -> Iterator[None]:
    """Name the surface a request came in on (`tui`, `eval`, `web`) for everything inside."""
    token = _current_channel.set(channel)
    try:
        yield
    finally:
        _current_channel.reset(token)


@contextmanager
def bound_log_context(
    *, agent: str, session: str, trace_id: str, channel: str | None = None
) -> Iterator[None]:
    """Stamp every log line inside this block with who, where and which trace.

    The channel is the caller's (`bound_channel`) unless given: the agent
    owns the turn, the surface that called it owns the channel.
    """
    values = {
        "agent": agent,
        "session": session,
        "channel": channel or _current_channel.get() or "",
        "trace_id": trace_id,
    }
    tokens: list[tuple[ContextVar[str | None], Token[str | None]]] = [
        (variable, variable.set(values[name])) for name, variable in _FIELDS
    ]
    try:
        yield
    finally:
        for variable, token in reversed(tokens):
            variable.reset(token)


def log_context_fields() -> dict[str, str]:
    """The bound identity fields, for a log record (only those that are bound)."""
    return {name: value for name, variable in _FIELDS if (value := variable.get())}


__all__ = [
    "bound_channel",
    "bound_log_context",
    "get_current_trace_id",
    "log_context_fields",
    "new_session_id",
    "new_trace_id",
    "set_current_trace_id",
]
