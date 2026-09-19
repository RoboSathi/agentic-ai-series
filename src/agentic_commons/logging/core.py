"""Structured, per-subsystem, colored logging (LLD §2).

The file log is the authoritative record and is written independently of the
TUI: the TUI's live feed is a human-readable summary only, so full structured
detail for the same event always lands here (PRD FR-56).
"""

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from rich.logging import RichHandler
from rich.text import Text

from agentic_commons.config import get_settings
from agentic_commons.logging.narration import NARRATION, Narration
from agentic_commons.logging.trace_context import get_current_trace_id

Level = Literal["debug", "info", "warning", "error"]

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
    "blast_radius",
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
    "design_patterns": "plum1",
    "episodes": "khaki1",
}


@runtime_checkable
class StructuredLogger(Protocol):
    """Emits structured events, optionally rebound to another subsystem."""

    def bind(self, subsystem: str) -> "StructuredLogger":
        ...

    def event(
        self,
        level: Level,
        message: str,
        **fields: Any,
    ) -> None:
        ...


class PlainFormatter(logging.Formatter):
    """Formatter that strips Rich markup for handlers that don't render it."""

    def format(
        self,
        record: logging.LogRecord,
    ) -> str:
        rendered = super().format(record)
        return Text.from_markup(rendered).plain


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

        suffix = (
            f" {json.dumps(fields, default=str)}"
            if fields
            else ""
        )

        self._logger.log(
            _LEVEL_TO_INT[level],
            f"{message}{suffix}",
            extra={
                NARRATION: Narration(
                    subsystem=self._subsystem,
                    headline=message,
                    fields=fields,
                )
            },
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

        suffix = (
            f" {json.dumps(fields, default=str)}"
            if fields
            else ""
        )

        self._logger.exception(
            f"{msg}{suffix}"
        )

    def log_called(
        self,
        func_name: str,
        color: str,
        arguments: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> None:
        """Trace a call at DEBUG.

        The file receives the full payload.
        The Narration receives safe summaries for the terminal.
        """

        all_args = ", ".join(
            f"{name}={_payload(value)}"
            for name, value in arguments.items()
        )

        self._traced(
            (
                f"[bold {color}]CALLED[/bold {color}] "
                f"[cyan]{func_name}[/cyan]({all_args})"
            ),
            Narration(
                subsystem=self._subsystem,
                kind="call",
                qualname=func_name,
                arguments={
                    name: _summarize(value)
                    for name, value in arguments.items()
                },
            ),
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
            Narration(
                subsystem=self._subsystem,
                kind="return",
                qualname=func_name,
                arguments={
                    name: _summarize(value)
                    for name, value in (arguments or {}).items()
                },
                outcome=_summarize(result),
                elapsed_ms=round(
                    elapsed_ms,
                    1,
                ),
            ),
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
            Narration(
                subsystem=self._subsystem,
                kind="skip",
                qualname=func_name,
                outcome=f"{toggle_path} is off",
            ),
        )

    def _traced(
        self,
        message: str,
        narration: "Narration",
    ) -> None:
        """One call-trace record."""

        self._logger.debug(
            message,
            extra={
                NARRATION: narration
            },
        )


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
            cls._loggers[subsystem] = cls._build(
                subsystem
            )

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
            for handler in list(
                logger.handlers
            ):
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

        logger = logging.getLogger(
            f"agentic.{subsystem}"
        )

        # Let DEBUG through the logger gate.
        # Each handler decides what it wants.
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        logger.handlers.clear()

        if settings.console_enabled:
            logger.addHandler(
                cls._console_handler(
                    settings.console_level
                )
            )

        if settings.file_enabled:
            logger.addHandler(
                cls._file_handler(
                    subsystem
                )
            )

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

        handler.setLevel(
            settings.file_level
        )

        handler.setFormatter(
            PlainFormatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )

        return handler


def get_logger(
    subsystem: str,
) -> RichLogger:
    """The logger for one subsystem — standard entry point."""
    return LoggerFactory.for_subsystem(
        subsystem
    )


def _summarize(
    value: Any,
) -> str:
    """Return a compact, safe summary of a traced value.

    Scalars speak for themselves.

    Collections become counts, except for small lists of strings/integers
    such as tool names or allowed actions.

    Structured objects expose several identifying fields so a return such as
    ``AuthorityDecision`` can show:

        action=...
        verdict=...
        reason=...

    in one trace line.

    Raw content remains in the file log.
    """

    if value is None or isinstance(
        value,
        bool,
    ):
        return str(value)

    if isinstance(
        value,
        (int, float),
    ):
        return str(value)

    if isinstance(
        value,
        str,
    ):
        return value

    if isinstance(
        value,
        dict,
    ):
        return (
            f"{len(value)} item"
            if len(value) == 1
            else f"{len(value)} items"
        )

    if isinstance(
        value,
        (list, tuple, set),
    ):
        if value and all(
            isinstance(
                item,
                (str, int),
            )
            for item in value
        ):
            return ", ".join(
                str(item)
                for item in value
            )

        return (
            f"{len(value)} item"
            if len(value) == 1
            else f"{len(value)} items"
        )

    details: list[str] = []

    for field in _IDENTIFYING_FIELDS:
        identity = getattr(
            value,
            field,
            None,
        )

        if isinstance(
            identity,
            bool,
        ):
            continue

        if identity in (
            None,
            "",
        ):
            continue

        if isinstance(
            identity,
            (str, int, float),
        ):
            details.append(
                f"{field}={identity}"
            )

        elif isinstance(
            identity,
            (list, tuple, set),
        ):
            if identity and all(
                isinstance(
                    item,
                    (str, int),
                )
                for item in identity
            ):
                rendered = ", ".join(
                    str(item)
                    for item in identity
                )

                details.append(
                    f"{field}=[{rendered}]"
                )

            else:
                details.append(
                    f"{field}="
                    f"{len(identity)} item"
                    if len(identity) == 1
                    else
                    f"{field}="
                    f"{len(identity)} items"
                )

        if len(details) >= 5:
            break

    # Many LLM result objects carry token usage under `.usage`.
    usage = getattr(
        value,
        "usage",
        None,
    )

    if usage is not None and len(details) < 5:
        total = getattr(
            usage,
            "total_tokens",
            None,
        )

        prompt = getattr(
            usage,
            "prompt_tokens",
            None,
        )

        completion = getattr(
            usage,
            "completion_tokens",
            None,
        )

        if total not in (
            None,
            0,
        ):
            details.append(
                f"tokens={total}"
            )

        elif (
            prompt not in (None, 0)
            or completion not in (None, 0)
        ):
            details.append(
                f"tokens={prompt or 0}+{completion or 0}"
            )

    return ", ".join(details)


def _payload(
    value: Any,
) -> str:
    """Render one traced value for the authoritative file log.

    The file log is allowed to retain the full payload subject to the
    configured logging payload limit.
    """

    rendered = repr(value)

    limit = get_settings().logging.payload_max_chars

    if (
        limit <= 0
        or len(rendered) <= limit
    ):
        return rendered

    return (
        f"{rendered[:limit]}… "
        f"[{len(rendered)} chars]"
    )