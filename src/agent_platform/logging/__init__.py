"""Logging module for the agent platform."""

from agent_platform.logging.core import (
    Level,
    LoggerFactory,
    PlainFormatter,
    RichLogger,
    StructuredLogger,
    get_logger,
    log_directory,
    shared_console,
    subsystem_color,
)
from agent_platform.logging.decorators import instrument, logged
from agent_platform.logging.errors import describe_error
from agent_platform.logging.trace_context import (
    bound_channel,
    bound_log_context,
    get_current_trace_id,
    log_context_fields,
    new_session_id,
    new_trace_id,
    set_current_trace_id,
)

__all__ = [
    "Level",
    "LoggerFactory",
    "PlainFormatter",
    "RichLogger",
    "StructuredLogger",
    "bound_channel",
    "bound_log_context",
    "describe_error",
    "get_current_trace_id",
    "get_logger",
    "instrument",
    "log_context_fields",
    "log_directory",
    "new_session_id",
    "new_trace_id",
    "logged",
    "set_current_trace_id",
    "shared_console",
    "subsystem_color",
]
