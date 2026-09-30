"""Structured, per-subsystem, colored logging (LLD §2).

The file log is the authoritative record and is written independently of the
TUI: the TUI's live feed is a human-readable summary only, so full structured
detail for the same event always lands here (PRD FR-56).
"""

import json
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from rich.logging import RichHandler
from rich.text import Text

from agentic_commons.config import get_settings
from agentic_commons.logging.trace_context import get_current_trace_id

Level = Literal["debug", "info", "warning", "error"]

_STRUCTURED_FIELDS = "_agentic_structured_fields"

_LEVEL_TO_INT: dict[Level, int] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}

# Fields that identify an object well enough to name it on screen, in
# the order they are tried.
#
# Everything else about the object remains in the file log.
_IDENTIFYING_FIELDS = (
    # Common business identifiers / decisions.
    "name",
    "action",
    "order_id",
    "customer_id",
    "key",
    "query",
    "verdict",
    "label",
    "topic",
    "route",
    # Decision / model detail that explains the step.
    "reason",
    "question",
    "decision",
    "amount",
    "requested_amount",
    "allowed_amount",
    "threshold",
    "policy_version",
    "finish_reason",
    "model",
    "total_tokens",
    "prompt_tokens",
    "completion_tokens",
    "tool_calls",
    "allowed_tools",
    "offered_actions",
    "surface",
    "reversible",
    "requires_authority_rank",
    "steps",
    "score",
)

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
    "research_agent": "light_goldenrod1",
    "audit_goldfish": "gold1",
    "scheduler": "medium_purple1",
    "budget": "salmon1",
    "web": "deep_sky_blue1",
    "ledger": "aquamarine1",
    "artifacts": "wheat1",
    "verification": "light_steel_blue",
    "design_patterns": "plum1",
    "episodes": "khaki1",
}


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


# Drawing characters for the field tree. A record's structure is the
# thing you are usually reading past in a log, so it is drawn rather
# than punctuated: the eye follows a line down a page faster than it
# counts braces.
_BRANCH = "├─ "
_LAST = "└─ "
_PIPE = "│  "
_GAP = "   "

# Long values are wrapped rather than cut. Full payloads are the point of
# these files — the screen is where things are summarised.
_WRAP_AT = 100


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
        return _tree(fields, prefix="  ")


def _file_format() -> str:
    """How file records are drawn, defaulting to the readable shape.

    Read defensively: the logger is initialised early, and a logging
    subsystem that raised because configuration was not ready yet would
    take down the thing it exists to explain.
    """
    try:
        from agentic_commons.config import get_settings

        return str(get_settings().logging.file_format)
    except Exception:
        return "tree"


def _tree(node: Any, prefix: str) -> list[str]:
    """One value, drawn as an indented tree."""
    if isinstance(node, dict):
        return _branches(list(node.items()), prefix, labelled=True)
    if isinstance(node, list):
        return _branches(list(enumerate(node)), prefix, labelled=False)
    return [f"{prefix}{line}" for line in _scalar_lines(node)]


def _branches(items: list[Any], prefix: str, labelled: bool) -> list[str]:
    """Each child of a mapping or sequence, with its connector."""
    lines: list[str] = []
    for index, (key, value) in enumerate(items):
        last = index == len(items) - 1
        connector = _LAST if last else _BRANCH
        label = f"{key}" if labelled else f"[{key}]"
        below = prefix + (_GAP if last else _PIPE)
        if isinstance(value, (dict, list)) and value:
            lines.append(f"{prefix}{connector}{label}:")
            lines.extend(_tree(value, below))
            continue
        rendered = _scalar_lines(value)
        lines.append(f"{prefix}{connector}{label}: {rendered[0]}")
        lines.extend(f"{below}{line}" for line in rendered[1:])
    return lines


def _scalar_lines(value: Any) -> list[str]:
    """A leaf value as one or more lines, never elided.

    Embedded newlines are honoured, because a system prompt is a document
    and reading it as one escaped string is how these files stopped being
    worth opening.
    """
    if value is None:
        return ["null"]
    if isinstance(value, bool):
        return ["true" if value else "false"]
    if isinstance(value, (dict, list)):
        return ["{}" if isinstance(value, dict) else "[]"]
    text = str(value)
    lines: list[str] = []
    for raw in text.splitlines() or [""]:
        lines.extend(_wrapped(raw))
    return lines or [""]


def _wrapped(line: str) -> list[str]:
    """One line, folded at a readable width rather than cut."""
    if len(line) <= _WRAP_AT:
        return [line]
    return [line[start : start + _WRAP_AT] for start in range(0, len(line), _WRAP_AT)]


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

        trace_id = get_current_trace_id()

        if trace_id:
            fields = {
                "trace_id": trace_id,
                **fields,
            }

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

        trace_id = get_current_trace_id()

        if trace_id:
            fields = {
                "trace_id": trace_id,
                **fields,
            }

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

        self._logger.debug(message, extra={_STRUCTURED_FIELDS: structured_fields or {}})


class LoggerFactory:
    """Builds one configured logger per subsystem (LLD §2)."""

    _loggers: dict[str, logging.Logger] = {}

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
            logger.addHandler(cls._console_handler(settings.console_level))

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
        handler = RichHandler(
            markup=True,
            rich_tracebacks=True,
            show_time=True,
            show_level=True,
            show_path=False,
        )

        handler.setLevel(level)

        handler.setFormatter(
            logging.Formatter(
                "%(message)s",
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

        log_dir: Path = settings.resolved_dir
        log_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        handler = RotatingFileHandler(
            log_dir / f"{subsystem}.log",
            maxBytes=settings.max_bytes,
            backupCount=settings.backup_count,
        )

        handler.setLevel(settings.file_level)

        handler.setFormatter(PlainFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

        return handler


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
