"""Call-tracing decorator (LLD §2, "Decorator" pattern).

`@logged` writes a CALLED/RETURNED pair at DEBUG level, with the arguments
and the return value in full (PRD FR-56). Passing `toggle_path` also makes
the call a no-op when that feature is switched off, returning a
type-appropriate placeholder instead; without it the wrapped function always
runs.

Async methods get an async wrapper. The distinction matters: a sync
wrapper around a coroutine function traces the coroutine *object* and
times how long it took to create — a line that looks like evidence and
contains none.
"""

import functools
import inspect
import time
import types
import typing
from typing import Any, Callable, TypeVar, cast

from agentic_commons.config import is_enabled
from agentic_commons.logging.core import SUBSYSTEM_COLORS, get_logger

T = TypeVar("T")
C = TypeVar("C")

# Set on a wrapper so `instrument` can tell an already-traced method from
# an untraced one and never wrap the same call twice.
_TRACED_MARKER = "__agentic_traced__"


def logged(
    subsystem: str, toggle_path: str | None = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Trace calls to a method, optionally gating it on a feature toggle.

    Args:
        subsystem: owning subsystem, used for the log file and color.
        toggle_path: dotted feature path (e.g. "security.pii"). When given
            and disabled, the wrapped call is skipped entirely.
    """
    color = SUBSYSTEM_COLORS.get(subsystem, "white")

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def announce(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, float]:
            """Log the call, or report that the toggle skipped it."""
            logger = get_logger(subsystem)
            if toggle_path is not None and not is_enabled(toggle_path):
                logger.log_skipped(func.__qualname__, color, toggle_path)
                return logger, -1.0
            visible_args = args[1:] if args and hasattr(args[0], "__dict__") else args
            logger.log_called(func.__qualname__, color, visible_args, kwargs)
            return logger, time.perf_counter()

        if inspect.iscoroutinefunction(func):
            # Awaited here rather than returned unawaited. A sync wrapper
            # around an async function traces the *coroutine object* and
            # reports 0.0ms, which is a trace of nothing: the call had not
            # run yet when the line was written.
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                logger, start = announce(args, kwargs)
                if start < 0:
                    return _placeholder_for(func)
                result = await func(*args, **kwargs)
                logger.log_returned(
                    func.__qualname__, color, result, (time.perf_counter() - start) * 1000
                )
                return result

            setattr(async_wrapper, _TRACED_MARKER, True)
            return cast(Callable[..., T], async_wrapper)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            logger, start = announce(args, kwargs)
            if start < 0:
                return _placeholder_for(func)
            result = func(*args, **kwargs)
            logger.log_returned(
                func.__qualname__, color, result, (time.perf_counter() - start) * 1000
            )
            return result

        setattr(wrapper, _TRACED_MARKER, True)
        return wrapper

    return decorator


def instrument(subsystem: str) -> Callable[[type[C]], type[C]]:
    """Trace every public method of a class (LLD §2).

    One annotation on the class instead of one per method, for the same
    reason a linter beats a checklist: the per-method version is correct
    only until somebody adds a method. A new step on a decorated class is
    traced because it exists, not because its author remembered.

    Left alone: dunders, private methods, properties, and anything
    already carrying `@logged` — double-tracing a method writes every
    call twice and makes the file log look like a loop.
    """

    def decorator(cls: type[C]) -> type[C]:
        for name, attribute in list(vars(cls).items()):
            if not _is_traceable(name, attribute):
                continue
            setattr(cls, name, logged(subsystem)(attribute))
        return cls

    return decorator


def _is_traceable(name: str, attribute: Any) -> bool:
    """Whether one class attribute is a method worth tracing."""
    if name.startswith("_"):
        return False
    if isinstance(attribute, (property, staticmethod, classmethod)):
        return False
    if getattr(attribute, _TRACED_MARKER, False):
        return False
    return inspect.isfunction(attribute)


def _placeholder_for(func: Callable[..., T]) -> T:
    """A type-appropriate empty value for a skipped (disabled) call.

    Never raises: a disabled subsystem must not crash its caller.
    """
    annotation = inspect.signature(func).return_annotation

    if annotation is inspect.Signature.empty or annotation is None or annotation is Any:
        return None  # type: ignore[return-value]
    simple: dict[Any, Any] = {bool: False, str: "", int: 0, float: 0.0}
    if annotation in simple:
        return simple[annotation]  # type: ignore[no-any-return]

    origin = typing.get_origin(annotation) or annotation
    if origin in (typing.Union, types.UnionType) and type(None) in typing.get_args(annotation):
        return None  # type: ignore[return-value]
    containers: dict[Any, Any] = {dict: {}, list: [], tuple: (), set: set()}
    if origin in containers:
        return containers[origin]  # type: ignore[no-any-return]

    if hasattr(annotation, "model_construct"):
        return annotation.model_construct()  # type: ignore[no-any-return]
    try:
        return annotation()  # type: ignore[no-any-return]
    except Exception:
        return None  # type: ignore[return-value]
