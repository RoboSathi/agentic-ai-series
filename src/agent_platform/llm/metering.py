"""Model usage, metered once for whoever is watching the current run.

Every provider call any agent makes passes through one adapter, and so does
its usage: the adapter reports it here, and whatever surface started the run
(the TUI's turn status, an eval, a script) binds a sink for its duration.
Agents never count tokens themselves, so no agent can forget to, and no two
reports of the same call can add up twice.

Bound with a token and reset on exit, like the trace context: a run's meter
ends with the run, and tasks it starts (a hedged backup, a parallel tool
call) inherit it.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token

from agent_platform.llm.types import TokenUsage

# A sink hears each call's whole usage: its tokens, and how many its provider served cached.
TokenSink = Callable[[TokenUsage], None]

# Every sink watching the current run: the surface that started it (the TUI's
# status bar) and the turn being summarised both hear each call.
_current_sinks: ContextVar[tuple[TokenSink, ...]] = ContextVar("current_token_sinks", default=())


@contextmanager
def metered(sink: TokenSink) -> Iterator[None]:
    """Send the tokens of every model call made inside this block to `sink`."""
    token = attach(sink)
    try:
        yield
    finally:
        detach(token)


def attach(sink: TokenSink) -> Token[tuple[TokenSink, ...]]:
    """Add `sink` to the current run's meters until `detach` (for a bracket, not a block)."""
    return _current_sinks.set((*_current_sinks.get(), sink))


def detach(token: Token[tuple[TokenSink, ...]]) -> None:
    """Undo one `attach`."""
    _current_sinks.reset(token)


def record_usage(usage: TokenUsage) -> None:
    """Report one call's usage to the run watching it, if any is."""
    if usage.total_tokens > 0:
        for sink in _current_sinks.get():
            sink(usage)


__all__ = ["TokenSink", "attach", "detach", "metered", "record_usage"]
