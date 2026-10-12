"""Structured, per-subsystem, colored logging (LLD §2).

The file log is the authoritative record and is written independently of the
TUI: the TUI's live feed is a human-readable summary only, so full structured
detail for the same event always lands here (PRD FR-56).
"""

import json
import logging
import re
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text
from rich.theme import Theme

from agent_platform.config import get_settings
from agent_platform.interface.presentation.text_tree import value_lines
from agent_platform.logging.trace_context import log_context_fields

Level = Literal["debug", "info", "warning", "error"]

_STRUCTURED_FIELDS = "_agentic_structured_fields"

_LEVEL_TO_INT: dict[Level, int] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}


def subsystem_color(subsystem: str) -> str:
    """The console colour a subsystem's call traces are drawn in (`logging.subsystem_colors`).

    Configured, not listed here: an agent's overlay adds its own subsystem,
    so the platform never has to name one.
    """
    settings = get_settings().logging
    return settings.subsystem_colors.get(subsystem, settings.subsystem_default_color)


@runtime_checkable
class StructuredLogger(Protocol):
    """Emits structured events, optionally rebound to another subsystem."""

    def bind(self, subsystem: str) -> "StructuredLogger": ...

    def event(
        self,
        level: Level,
        message: str,
        **fields: Any,
    ) -> None: ...


class PlainFormatter(logging.Formatter):
    """Human-readable structured formatter for authoritative subsystem logs.

    Two shapes, chosen by `logging.file_format`. `tree` draws a record's
    fields as an indented tree; `json` emits the pretty-printed object,
    for anything that parses these files rather than reads them.

    The tree exists because the JSON did not survive contact with a real
    payload. A logged LLM call is a system prompt, a transcript and a
    completion, and as `json.dumps(indent=2)` that is a wall of escaped
    newlines and quotes in which the one line you wanted is invisible.
    Nothing is truncated in either shape.
    """

    def format(self, record: logging.LogRecord) -> str:
        timestamp = (
            datetime.fromtimestamp(record.created)
            .astimezone()
            .strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        )
        module = record.name.removeprefix("agentic.")
        level = record.levelname
        message = Text.from_markup(record.getMessage()).plain
        fields = getattr(record, _STRUCTURED_FIELDS, {}) or {}

        lines = [f"{timestamp} | {module} | {level:<7} | {message}"]
        if fields:
            lines.extend(self._render(_json_safe(fields)))
        if record.exc_info:
            lines.append(self.formatException(record.exc_info))
        return "\n".join(lines)

    def _render(self, fields: dict[str, Any]) -> list[str]:
        """The record's fields, in whichever shape is configured."""
        if _file_format() == "json":
            rendered = json.dumps(fields, indent=2, ensure_ascii=False, default=str)
            return [f"  {line}" for line in rendered.splitlines()]
        return value_lines(fields, prefix="  ")


def _file_format() -> str:
    """How file records are drawn, defaulting to the readable shape.

    Read defensively: the logger is initialised early, and a logging
    subsystem that raised because configuration was not ready yet would
    take down the thing it exists to explain.
    """
    try:
        from agent_platform.config import get_settings

        return str(get_settings().logging.file_format)
    except Exception:
        return "tree"


class RichLogger:
    """StructuredLogger over stdlib logging, with Rich console rendering."""

    def __init__(
        self,
        logger: logging.Logger,
        subsystem: str,
    ) -> None:
        self._logger = logger
        self._subsystem = subsystem

    @property
    def subsystem(self) -> str:
        return self._subsystem

    def bind(
        self,
        subsystem: str,
    ) -> "RichLogger":
        """A logger for another subsystem, sharing this configuration."""
        return get_logger(subsystem)

    def event(
        self,
        level: Level,
        message: str,
        **fields: Any,
    ) -> None:
        """Log one structured event."""

        fields = {**log_context_fields(), **fields}

        self._logger.log(
            _LEVEL_TO_INT[level],
            message,
            extra={_STRUCTURED_FIELDS: fields},
        )

    def info(
        self,
        msg: str,
        **fields: Any,
    ) -> None:
        self.event(
            "info",
            msg,
            **fields,
        )

    def debug(
        self,
        msg: str,
        **fields: Any,
    ) -> None:
        self.event(
            "debug",
            msg,
            **fields,
        )

    def warning(
        self,
        msg: str,
        **fields: Any,
    ) -> None:
        self.event(
            "warning",
            msg,
            **fields,
        )

    def error(
        self,
        msg: str,
        **fields: Any,
    ) -> None:
        self.event(
            "error",
            msg,
            **fields,
        )

    def exception(
        self,
        msg: str,
        **fields: Any,
    ) -> None:
        """Log at ERROR with the active traceback attached."""

        fields = {**log_context_fields(), **fields}

        self._logger.exception(
            msg,
            extra={
                _STRUCTURED_FIELDS: fields,
            },
        )

    def log_called(
        self,
        func_name: str,
        color: str,
        arguments: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> None:
        """Trace a call at DEBUG — the file receives the full payload."""

        all_args = ", ".join(f"{name}={_payload(value)}" for name, value in arguments.items())

        self._traced(
            (f"[bold {color}]CALLED[/bold {color}] [cyan]{func_name}[/cyan]({all_args})"),
            structured_fields={
                "function": func_name,
                "arguments": arguments,
            },
        )

    def log_returned(
        self,
        func_name: str,
        color: str,
        result: Any,
        elapsed_ms: float,
        arguments: dict[str, Any] | None = None,
    ) -> None:
        """Trace a return value and latency at DEBUG."""

        self._traced(
            (
                f"[bold {color}]RETURNED[/bold {color}] "
                f"[cyan]{func_name}[/cyan] → "
                f"{_payload(result)} "
                f"[dim]({elapsed_ms:.1f}ms)[/dim]"
            ),
            structured_fields={
                "function": func_name,
                "arguments": arguments or {},
                "result": result,
                "elapsed_ms": round(elapsed_ms, 1),
            },
        )

    def log_skipped(
        self,
        func_name: str,
        color: str,
        toggle_path: str,
    ) -> None:
        """Trace a subsystem skipped because its toggle is off."""

        self._traced(
            (
                f"[dim {color}]SKIPPED[/dim {color}] "
                f"[cyan]{func_name}[/cyan] "
                f"[yellow]({toggle_path}=False)[/yellow]"
            ),
        )

    def _traced(
        self,
        message: str,
        structured_fields: dict[str, Any] | None = None,
    ) -> None:
        """One call-trace record with machine-readable file details."""

        fields = {**log_context_fields(), **(structured_fields or {})}
        self._logger.debug(message, extra={_STRUCTURED_FIELDS: fields})


_SHARED_CONSOLE: Console | None = None


def shared_console() -> Console:
    """The one console every log handler writes to.

    One console, so a live display on it (an eval's progress bar) is redrawn around log
    lines instead of being written over by them.
    """
    global _SHARED_CONSOLE
    if _SHARED_CONSOLE is None:
        _SHARED_CONSOLE = Console(theme=Theme(get_settings().logging.console_theme))
    return _SHARED_CONSOLE


class LoggerFactory:
    """Builds one configured logger per subsystem (LLD §2)."""

    _loggers: dict[str, logging.Logger] = {}
    # A process-wide console level that replaces `logging.console_level` (None: the config's).
    _console_level: str | None = None

    # Handlers that want every subsystem's records, now and in future.
    _subscribers: list[logging.Handler] = []

    @classmethod
    def for_subsystem(
        cls,
        subsystem: str,
    ) -> RichLogger:
        """Get or build the logger for one subsystem."""

        if subsystem not in cls._loggers:
            cls._loggers[subsystem] = cls._build(subsystem)

        return RichLogger(
            cls._loggers[subsystem],
            subsystem,
        )

    @classmethod
    def subscribe(
        cls,
        handler: logging.Handler,
    ) -> None:
        """Send every subsystem's records to a subscriber."""

        if handler in cls._subscribers:
            return

        cls._subscribers.append(handler)

        for logger in cls._loggers.values():
            logger.addHandler(handler)

    @classmethod
    def log_everything_to(cls, name: str) -> None:
        """Also write every subsystem's records to one file, `<name>.log` in the agent's folder.

        For a run whose console is a progress display: the detail stays readable in one file
        instead of scrolling past the bar. Calling it again with the same name is a no-op.
        """
        sink = cls._file_handler(name)
        if any(
            getattr(h, "baseFilename", None) == getattr(sink, "baseFilename", "")
            for h in cls._subscribers
        ):
            sink.close()
            return
        cls.subscribe(sink)

    @classmethod
    def unsubscribe(
        cls,
        handler: logging.Handler,
    ) -> None:
        """Stop sending records to a subscriber."""

        if handler in cls._subscribers:
            cls._subscribers.remove(handler)

        for logger in cls._loggers.values():
            logger.removeHandler(handler)

    @classmethod
    def set_console_level(cls, level: str) -> None:
        """Change what reaches the console, for every subsystem now and later (files keep all)."""
        cls._console_level = level
        for logger in cls._loggers.values():
            for handler in logger.handlers:
                if isinstance(handler, RichHandler):
                    handler.setLevel(level)

    @classmethod
    def reset(cls) -> None:
        """Drop cached loggers so config changes take effect."""

        for logger in cls._loggers.values():
            for handler in list(logger.handlers):
                if handler in cls._subscribers:
                    logger.removeHandler(handler)
                    continue

                handler.close()
                logger.removeHandler(handler)

        cls._loggers.clear()

    @classmethod
    def _build(
        cls,
        subsystem: str,
    ) -> logging.Logger:
        settings = get_settings().logging

        logger = logging.getLogger(f"agentic.{subsystem}")

        # Let DEBUG through the logger gate.
        # Each handler decides what it wants.
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        logger.handlers.clear()

        if settings.console_enabled:
            logger.addHandler(cls._console_handler(cls._console_level or settings.console_level))

        if settings.file_enabled:
            logger.addHandler(cls._file_handler(subsystem))

        # IMPORTANT:
        # A TUI trace subscriber may have been attached before a subsystem's
        # logger was created. In that case the subscriber is still attached.
        for subscriber in cls._subscribers:
            logger.addHandler(subscriber)

        return logger

    @classmethod
    def _console_handler(
        cls,
        level: str,
    ) -> logging.Handler:
        settings = get_settings().logging
        handler = RichHandler(
            console=shared_console(),
            markup=True,
            rich_tracebacks=True,
            show_time=True,
            show_level=True,
            show_path=False,
        )

        handler.setLevel(level)

        handler.setFormatter(
            logging.Formatter(
                settings.console_format,
                datefmt="[%X]",
            )
        )

        return handler

    @classmethod
    def _file_handler(
        cls,
        subsystem: str,
    ) -> logging.Handler:
        settings = get_settings().logging

        handler = RotatingFileHandler(
            log_directory() / f"{subsystem}.log",
            maxBytes=settings.max_bytes,
            backupCount=settings.backup_count,
        )

        handler.setLevel(settings.file_level)

        handler.setFormatter(PlainFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

        return handler


_UNSAFE_FOLDER_CHARS = re.compile(r"[^\w.-]+")


def log_directory() -> Path:
    """Where this process's log files go: `<logging.dir>/<agent id>[/<run_label>]`, created.

    Each agent owns a folder, so two agents (or an agent and its offline eval runs, via
    `logging.run_label`) never write to the same file. The platform does not name an agent:
    the id is whatever the running agent's overlay declares.
    """
    settings = get_settings()
    parts = [settings.agent.id, settings.logging.run_label]
    folder = settings.logging.resolved_dir.joinpath(
        *(_UNSAFE_FOLDER_CHARS.sub("_", part) for part in parts if part)
    )
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def get_logger(
    subsystem: str,
) -> RichLogger:
    """The logger for one subsystem — standard entry point."""
    return LoggerFactory.for_subsystem(subsystem)


def _json_safe(value: Any) -> Any:
    """Convert structured log payloads to readable JSON without object addresses."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return _json_safe(value.model_dump(mode="json"))
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return {
                str(k): _json_safe(v) for k, v in vars(value).items() if not str(k).startswith("_")
            }
        except Exception:
            pass
    return str(value)


def _payload(
    value: Any,
) -> str:
    """Render one traced value for the authoritative file log.

    The file log is allowed to retain the full payload subject to the
    configured logging payload limit.
    """

    rendered = repr(value)

    limit = get_settings().logging.payload_max_chars

    if limit <= 0 or len(rendered) <= limit:
        return rendered

    return f"{rendered[:limit]}… [{len(rendered)} chars]"
