"""Real logging implementation using Rich and Python's logging."""

import logging
from pathlib import Path
from typing import Any

from rich.logging import RichHandler

from agentic_commons.config import get_settings

_loggers: dict[str, logging.Logger] = {}


def get_logger(subsystem: str) -> "RichLogger":
    """Get or create a logger for a subsystem."""
    if subsystem not in _loggers:
        settings = get_settings()
        log_dir = Path(settings.logging.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger(f"agentic.{subsystem}")
        logger.setLevel(settings.logging.level)
        logger.handlers.clear()

        if settings.logging.console_enabled:
            console_handler = RichHandler(
                markup=True,
                rich_tracebacks=True,
                show_time=True,
                show_level=True,
                show_path=False,
            )
            console_handler.setFormatter(
                logging.Formatter("%(message)s", datefmt="[%X]")
            )
            logger.addHandler(console_handler)

        if settings.logging.file_enabled:
            log_file = log_dir / f"{subsystem}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            logger.addHandler(file_handler)

        _loggers[subsystem] = logger

    return RichLogger(_loggers[subsystem], subsystem)


class RichLogger:
    """Wrapper around standard logger with rich formatting methods."""

    def __init__(self, logger: logging.Logger, subsystem: str) -> None:
        self._logger = logger
        self._subsystem = subsystem

    def log_called(
        self,
        func_name: str,
        color: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        """Log a function call with subsystem color."""
        args_repr = ", ".join(repr(a)[:50] for a in args)
        kwargs_repr = ", ".join(
            f"{k}={repr(v)[:50]}" for k, v in kwargs.items()
        )
        all_args = ", ".join(filter(None, [args_repr, kwargs_repr]))
        msg = (
            f"[bold {color}]CALLED[/bold {color}] "
            f"[cyan]{func_name}[/cyan]({all_args})"
        )
        self._logger.info(msg)

    def log_returned(
        self,
        func_name: str,
        color: str,
        result: Any,
        elapsed_ms: float,
    ) -> None:
        """Log a function return with result and timing."""
        result_repr = repr(result)[:100]
        msg = (
            f"[bold {color}]RETURNED[/bold {color}] "
            f"[cyan]{func_name}[/cyan] → {result_repr} "
            f"[dim]({elapsed_ms:.1f}ms)[/dim]"
        )
        self._logger.info(msg)

    def log_skipped(
        self,
        func_name: str,
        color: str,
        toggle_path: str,
    ) -> None:
        """Log a subsystem skip due to disabled toggle."""
        msg = (
            f"[dim {color}]SKIPPED[/dim {color}] "
            f"[cyan]{func_name}[/cyan] "
            f"[yellow]({toggle_path}=False)[/yellow]"
        )
        self._logger.info(msg)

    def info(self, msg: str) -> None:
        """Log info message."""
        self._logger.info(msg)

    def debug(self, msg: str) -> None:
        """Log debug message."""
        self._logger.debug(msg)

    def warning(self, msg: str) -> None:
        """Log warning message."""
        self._logger.warning(msg)

    def error(self, msg: str) -> None:
        """Log error message."""
        self._logger.error(msg)
