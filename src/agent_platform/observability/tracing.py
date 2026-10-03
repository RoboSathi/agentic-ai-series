"""Stage tracing as a decorator (LLD §6, "Decorator" pattern).

`@traced` puts one method on the record twice over: a structured
CALLED/RETURNED pair in the subsystem's log file, and — when the object
owns a `StageReporter` — a started/completed pair on the live feed the
TUI renders. One annotation, both audiences.

It lives in `observability` rather than `tui` on purpose. Business code
that imported the UI package to announce its own progress would invert
the layering: the TUI is a *consumer* of this stream, and the graph must
not know whether anything is watching. (An earlier draft of this
decorator lived under `tui/` and was imported by the intent classifier,
which made `customer_support_agent` depend on Textual to classify a
sentence.)

Failures are re-raised, always. A tracing decorator that swallows an
exception to keep the feed tidy is a tracing decorator that hides the
one event worth tracing.
"""

import functools
import inspect
import time
from typing import Any, Callable, Literal, TypeVar, cast

from agent_platform.logging import get_logger
from agent_platform.logging.decorators import bind_arguments
from agent_platform.observability.status_feed import StageReporter

F = TypeVar("F", bound=Callable[..., Any])

# Attributes a traced object may expose. Duck-typed rather than required
# by a base class: a class should not have to inherit anything to be
# observable, and a component with no reporter simply logs.
_REPORTER_ATTR = "_reporter"
_LOGGER_ATTR = "_logger"

# Result fields worth putting in the feed's one-line detail, in the order
# they are tried. A stage's detail should say *what it decided*, and
# these are the names the domain already uses for that.
# Traced stages are steps in the agent's own flow, so they are coloured
# like the component that owns them rather than given a colour of their
# own — the subsystem is what a reader follows down the feed.
_TRACE_COLOR = "cyan"

_DETAIL_FIELDS = ("name", "detail", "reason", "verdict", "route", "topic", "matched_phrase")


def traced(stage: str | None = None, subsystem: str = "observability") -> Callable[[F], F]:
    """Log a method's lifecycle, and report it to the live feed.

    Args:
        stage: canonical stage key for the feed. Defaults to the method
            name, which is usually already the name of the step.
        subsystem: which log file the trace lands in when the object has
            no logger of its own.

    Works on sync and async methods alike: the wrapper is chosen once, at
    decoration time, so an async method is never wrapped in a sync shim
    that returns an un-awaited coroutine.
    """

    def decorator(func: F) -> F:
        stage_name = stage or func.__name__

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
                with _Trace(self, stage_name, subsystem, func, args, kwargs) as trace:
                    return trace.finish(await func(self, *args, **kwargs))

            return cast(F, async_wrapper)

        @functools.wraps(func)
        def sync_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            with _Trace(self, stage_name, subsystem, func, args, kwargs) as trace:
                return trace.finish(func(self, *args, **kwargs))

        return cast(F, sync_wrapper)

    return decorator


class _Trace:
    """One traced execution: announce, time, and close out.

    A context manager so that the sync and async wrappers share every
    rule about what gets logged and reported — including the failure
    path, which is the one that must not differ between them.
    """

    def __init__(
        self,
        owner: Any,
        stage: str,
        subsystem: str,
        func: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        self._stage = stage
        self._logger = getattr(owner, _LOGGER_ATTR, None) or get_logger(subsystem)
        reporter = getattr(owner, _REPORTER_ATTR, None)
        self._reporter = reporter if isinstance(reporter, StageReporter) else None
        self._func = func
        self._qualname = func.__qualname__
        self._args = args
        self._kwargs = kwargs
        self._started = 0.0

    def __enter__(self) -> "_Trace":
        self._started = time.perf_counter()
        self._logger.log_called(
            self._qualname, _TRACE_COLOR, bind_arguments(self._func, self._args, self._kwargs), {}
        )
        if self._reporter is not None:
            self._reporter.started(self._stage)
        return self

    def finish(self, result: Any) -> Any:
        """Record a successful return, and pass the value straight through."""
        detail = _detail_of(result)
        self._logger.log_returned(
            self._qualname,
            _TRACE_COLOR,
            result,
            self._elapsed_ms(),
            bind_arguments(self._func, self._args, self._kwargs),
        )
        if self._reporter is not None:
            self._reporter.completed(self._stage, detail=detail)
        return result

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> Literal[False]:
        """Close out a failure. Never suppresses the exception."""
        if exc is None:
            return False
        self._logger.error(
            f"{self._stage} failed",
            stage=self._stage,
            elapsed_ms=self._elapsed_ms(),
            error=str(exc),
            error_type=type(exc).__name__,
        )
        if self._reporter is not None:
            self._reporter.failed(self._stage, detail=str(exc))
        return False

    def _elapsed_ms(self) -> float:
        return (time.perf_counter() - self._started) * 1000


def _detail_of(result: Any) -> str | None:
    """The one phrase from a result worth showing beside the stage.

    Reads the first of a handful of conventional field names. Anything
    else is left to the log file: a feed line that tries to summarise an
    arbitrary object ends up printing its `repr`, which is not a summary.
    """
    for field in _DETAIL_FIELDS:
        value = getattr(result, field, None)
        if isinstance(value, str) and value:
            return value
    return None
