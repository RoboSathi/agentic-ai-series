"""Structured, per-subsystem, colored logging (LLD §2).

The file log is the authoritative record and is written independently of the
TUI: the TUI's live feed is a human-readable summary only, so full
structured detail for the same event always lands here (PRD FR-56).
"""

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from rich.logging import RichHandler
from rich.text import Text

from agentic_commons.config import get_settings
from agentic_commons.logging.trace_context import get_current_trace_id

Level = Literal["debug", "info", "warning", "error"]

_LEVEL_TO_INT: dict[Level, int] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}

SUBSYSTEM_COLORS: dict[str, str] = {
    "config": "grey58",
    "logging": "grey78",
    "tui": "bright_magenta",
    "llm": "bright_blue",
    "persistence": "blue",
    "observability": "cyan",
    "audit": "green",
    "evals": "spring_green2",
    "guardrails": "magenta",
    "resilience": "yellow",
    "security": "red",
    "memory": "dodger_blue1",
    "context": "turquoise2",
    "tools": "white",
    "gatekeeper": "bright_red",
    "hallucination_protection": "orange3",
    "customer_support_agent": "bright_green",
    "design_patterns": "plum1",
    "episodes": "khaki1",
}


@runtime_checkable
class StructuredLogger(Protocol):
    """Emits structured events, optionally rebound to another subsystem."""

    def bind(self, subsystem: str) -> "StructuredLogger": ...

    def event(self, level: Level, message: str, **fields: Any) -> None: ...


class PlainFormatter(logging.Formatter):
    """Formatter that strips Rich markup for handlers that don't render it.

    Messages carry Rich markup (e.g. "[bright_green]…[/bright_green]") for
    the console; a plain file handler would otherwise write the literal tags.
    """

    def format(self, record: logging.LogRecord) -> str:
        rendered = super().format(record)
        return Text.from_markup(rendered).plain


class RichLogger:
    """StructuredLogger over stdlib logging, with Rich console rendering."""

    def __init__(self, logger: logging.Logger, subsystem: str) -> None:
        self._logger = logger
        self._subsystem = subsystem

    @property
    def subsystem(self) -> str:
        return self._subsystem

    def bind(self, subsystem: str) -> "RichLogger":
        """A logger for another subsystem, sharing this one's configuration."""
        return get_logger(subsystem)

    def event(self, level: Level, message: str, **fields: Any) -> None:
        """Log one structured event; extra fields are serialized as JSON."""
        trace_id = get_current_trace_id()
        if trace_id:
            fields = {"trace_id": trace_id, **fields}
        suffix = f" {json.dumps(fields, default=str)}" if fields else ""
        self._logger.log(_LEVEL_TO_INT[level], f"{message}{suffix}")

    def info(self, msg: str, **fields: Any) -> None:
        self.event("info", msg, **fields)

    def debug(self, msg: str, **fields: Any) -> None:
        self.event("debug", msg, **fields)

    def warning(self, msg: str, **fields: Any) -> None:
        self.event("warning", msg, **fields)

    def error(self, msg: str, **fields: Any) -> None:
        self.event("error", msg, **fields)

    def exception(self, msg: str, **fields: Any) -> None:
        """Log at ERROR with the active traceback attached.

        Only meaningful inside an `except` block — the traceback is what
        makes a swallowed exception diagnosable after the fact.
        """
        trace_id = get_current_trace_id()
        if trace_id:
            fields = {"trace_id": trace_id, **fields}
        suffix = f" {json.dumps(fields, default=str)}" if fields else ""
        self._logger.exception(f"{msg}{suffix}")

    def log_called(
        self, func_name: str, color: str, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        """Trace a call at DEBUG, with its arguments in full."""
        args_repr = ", ".join(_payload(a) for a in args)
        kwargs_repr = ", ".join(f"{k}={_payload(v)}" for k, v in kwargs.items())
        all_args = ", ".join(filter(None, [args_repr, kwargs_repr]))
        self.debug(f"[bold {color}]CALLED[/bold {color}] [cyan]{func_name}[/cyan]({all_args})")

    def log_returned(self, func_name: str, color: str, result: Any, elapsed_ms: float) -> None:
        """Trace a return value and its latency at DEBUG, in full."""
        self.debug(
            f"[bold {color}]RETURNED[/bold {color}] [cyan]{func_name}[/cyan] → "
            f"{_payload(result)} [dim]({elapsed_ms:.1f}ms)[/dim]"
        )

    def log_skipped(self, func_name: str, color: str, toggle_path: str) -> None:
        """Trace a subsystem skipped because its toggle is off."""
        self.debug(
            f"[dim {color}]SKIPPED[/dim {color}] [cyan]{func_name}[/cyan] "
            f"[yellow]({toggle_path}=False)[/yellow]"
        )


class LoggerFactory:
    """Builds one configured logger per subsystem (LLD §2)."""

    _loggers: dict[str, logging.Logger] = {}
    # Handlers that want every subsystem's records, whenever that
    # subsystem's logger happens to be built. The TUI's trace feed is one:
    # it subscribes once at startup, and must still receive records from a
    # subsystem whose logger is created twenty stages later.
    _subscribers: list[logging.Handler] = []

    @classmethod
    def for_subsystem(cls, subsystem: str) -> RichLogger:
        """Get (or build) the logger for one subsystem."""
        if subsystem not in cls._loggers:
            cls._loggers[subsystem] = cls._build(subsystem)
        return RichLogger(cls._loggers[subsystem], subsystem)

    @classmethod
    def subscribe(cls, handler: logging.Handler) -> None:
        """Send every subsystem's records to `handler`, now and in future.

        Attached to the loggers that already exist *and* remembered for
        the ones that do not. A subscriber that only saw today's loggers
        would silently miss whichever subsystem happened to start late.
        """
        if handler in cls._subscribers:
            return
        cls._subscribers.append(handler)
        for logger in cls._loggers.values():
            logger.addHandler(handler)

    @classmethod
    def unsubscribe(cls, handler: logging.Handler) -> None:
        """Stop sending records to `handler`."""
        if handler in cls._subscribers:
            cls._subscribers.remove(handler)
        for logger in cls._loggers.values():
            logger.removeHandler(handler)

    @classmethod
    def reset(cls) -> None:
        """Drop cached loggers so config changes take effect (tests use this)."""
        for logger in cls._loggers.values():
            for handler in list(logger.handlers):
                if handler in cls._subscribers:
                    logger.removeHandler(handler)
                    continue
                handler.close()
                logger.removeHandler(handler)
        cls._loggers.clear()

    @classmethod
    def _build(cls, subsystem: str) -> logging.Logger:
        settings = get_settings().logging
        logger = logging.getLogger(f"agentic.{subsystem}")
        # Let DEBUG through the logger gate; each handler decides who sees it.
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        logger.handlers.clear()
        if settings.console_enabled:
            logger.addHandler(cls._console_handler(settings.console_level))
        if settings.file_enabled:
            logger.addHandler(cls._file_handler(subsystem))
        for subscriber in cls._subscribers:
            logger.addHandler(subscriber)
        return logger

    @classmethod
    def _console_handler(cls, level: str) -> logging.Handler:
        handler = RichHandler(
            markup=True, rich_tracebacks=True, show_time=True, show_level=True, show_path=False
        )
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
        return handler

    @classmethod
    def _file_handler(cls, subsystem: str) -> logging.Handler:
        settings = get_settings().logging
        log_dir: Path = settings.resolved_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            log_dir / f"{subsystem}.log",
            maxBytes=settings.max_bytes,
            backupCount=settings.backup_count,
        )
        handler.setLevel(settings.file_level)
        handler.setFormatter(PlainFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        return handler


def get_logger(subsystem: str) -> RichLogger:
    """The logger for one subsystem — the standard entry point."""
    return LoggerFactory.for_subsystem(subsystem)


def _payload(value: Any) -> str:
    """One traced value, rendered at the configured length.

    The default is no limit at all. A truncated trace is the one that
    fails you: the argument you need to see is always the long one, and
    an elided payload turns the log from evidence into a hint.
    """
    rendered = repr(value)
    limit = get_settings().logging.payload_max_chars
    if limit <= 0 or len(rendered) <= limit:
        return rendered
    return f"{rendered[:limit]}… [{len(rendered)} chars]"
