"""Logging module for the agent platform."""

from agent_platform.logging.core import (
    SUBSYSTEM_COLORS,
    Level,
    LoggerFactory,
    PlainFormatter,
    RichLogger,
    StructuredLogger,
    get_logger,
    log_directory,
)
from agent_platform.logging.decorators import instrument, logged
from agent_platform.logging.errors import describe_error
from agent_platform.logging.trace_context import get_current_trace_id, set_current_trace_id

__all__ = [
    "Level",
    "LoggerFactory",
    "PlainFormatter",
    "RichLogger",
    "SUBSYSTEM_COLORS",
    "StructuredLogger",
    "describe_error",
    "get_current_trace_id",
    "get_logger",
    "instrument",
    "log_directory",
    "logged",
    "set_current_trace_id",
]
