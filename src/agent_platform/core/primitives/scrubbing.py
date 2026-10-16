"""The pass every writer makes before it writes: one function, installed once, applied everywhere.

The logger and the status feed sit below the security layer that knows what personal data is,
so they cannot import it. Instead the security layer installs its pass here when an agent is
composed (`security.pii.active.install_pii_scrubbing`), and each writer calls `scrub` on what
it is about to write. Nothing installed: the value is written as it is.

A pass that itself writes (a vault insert that logs) would call back into the logger and so
into the pass: the re-entrant call returns its value untouched rather than recursing.
"""

from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

Scrubber = Callable[[Any], Any]
# Told of the rows a query read, before anything — the read's own log line included — holds them.
RowObserver = Callable[[str, list[dict[str, Any]]], None]

_installed: list[Scrubber] = []
_row_observers: list[RowObserver] = []
_inside: ContextVar[bool] = ContextVar("inside_scrubber", default=False)


def install_scrubber(scrubber: Scrubber) -> None:
    """Make `scrubber` the pass every writer applies (replacing any earlier one)."""
    _installed[:] = [scrubber]


def install_row_observer(observer: RowObserver) -> None:
    """Make `observer` the one told of every row the database layer reads."""
    _row_observers[:] = [observer]


def observe_rows(query: str, rows: list[dict[str, Any]]) -> None:
    """Tell the installed observer what `query` read (nothing installed: nothing to tell)."""
    if _row_observers and rows:
        _row_observers[0](query, rows)


def scrub(value: Any) -> Any:
    """`value` as it may be written: through the installed pass, or as it is without one."""
    if not _installed or _inside.get():
        return value
    token = _inside.set(True)
    try:
        return _installed[0](value)
    finally:
        _inside.reset(token)


__all__ = [
    "RowObserver",
    "Scrubber",
    "install_row_observer",
    "install_scrubber",
    "observe_rows",
    "scrub",
]
