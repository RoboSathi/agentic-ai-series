"""Every log record, on screen (LLD §20).

The curated stage feed says what the agent is *doing*. This says what the
code is *executing*: one line per log record, from every subsystem, at
whatever level is configured — the same records that land in `logs/`,
mirrored to the panel while they happen.

Two streams rather than one, because they answer different questions. An
operator watching a customer conversation wants eight readable stages; a
developer tracking down why the third tool call never fired wants the
four hundred lines underneath them. The trace stream can be turned off at
runtime (`/debug off`) without touching the files, which stay complete
either way.

The handler is deliberately dumb and total: it formats and forwards, and
swallows nothing except its own failures. A UI that can break logging is
worse than no UI.
"""

import logging
from typing import Callable

from rich.text import Text

from agentic_commons.config import get_settings
from agentic_commons.logging.core import SUBSYSTEM_COLORS, LoggerFactory

# Where a record's subsystem is recoverable from: loggers are named
# `agentic.<subsystem>`, which is what the factory builds them as.
_LOGGER_PREFIX = "agentic."

_LEVEL_STYLES = {
    logging.DEBUG: "bright_black",
    logging.INFO: "white",
    logging.WARNING: "yellow",
    logging.ERROR: "red",
    logging.CRITICAL: "bold red",
}


class LiveFeedLogHandler(logging.Handler):
    """Forwards log records to a callback, as styled lines.

    Attached to every subsystem logger through `LoggerFactory.subscribe`,
    so a subsystem whose logger is built later still reaches the screen.
    """

    def __init__(self, sink: Callable[[Text], None], level: int = logging.DEBUG) -> None:
        super().__init__(level=level)
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        """Render one record and hand it to the panel.

        `logging.Handler.emit` must never raise — a failure here would
        surface as an exception inside whatever code was merely logging.
        """
        try:
            self._sink(self.render(record))
        except Exception:  # pragma: no cover - defensive, per logging contract
            self.handleError(record)

    @staticmethod
    def render(record: logging.LogRecord) -> Text:
        """`  memory · Evicted from working memory {"key": "order_10001"}`.

        The subsystem keeps its configured colour, so the eye can follow
        one module down the feed; the message keeps the markup it was
        written with, so a CALLED/RETURNED pair reads the same on screen
        as in the file.
        """
        subsystem = record.name.removeprefix(_LOGGER_PREFIX)
        line = Text("  ")
        line.append(f"{subsystem:<24}", style=SUBSYSTEM_COLORS.get(subsystem, "white"))
        line.append(" ")
        body = Text.from_markup(record.getMessage())
        body.stylize(_LEVEL_STYLES.get(record.levelno, "white"))
        return line + body


class TraceStream:
    """Turns the trace feed on and off for one console.

    Owns the subscription so the app does not have to remember whether it
    is currently attached — `enable()` twice is the same as once, which
    matters when the toggle is a command an operator can hammer.
    """

    def __init__(self, sink: Callable[[Text], None]) -> None:
        self._sink = sink
        self._handler: LiveFeedLogHandler | None = None

    @property
    def enabled(self) -> bool:
        return self._handler is not None

    @property
    def level(self) -> str:
        """The configured level, as a name."""
        return get_settings().tui.trace.level.upper()

    def enable(self, level: str | None = None) -> None:
        """Start mirroring log records to the panel."""
        self.disable()
        wanted = (level or self.level).upper()
        self._handler = LiveFeedLogHandler(self._sink, level=logging.getLevelNamesMapping()[wanted])
        LoggerFactory.subscribe(self._handler)

    def disable(self) -> None:
        """Stop mirroring. The files keep every record either way."""
        if self._handler is not None:
            LoggerFactory.unsubscribe(self._handler)
            self._handler = None
