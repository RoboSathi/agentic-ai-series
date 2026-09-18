"""Call-tracing decorator (LLD §2, "Decorator" pattern).

`@logged` writes a CALLED/RETURNED pair at DEBUG level — file only, so the
TUI's curated feed stays readable (PRD FR-56). Passing `toggle_path` also
makes the call a no-op when that feature is switched off, returning a
type-appropriate placeholder instead; without it the wrapped function always
runs.
"""

import functools
import inspect
import time
import types
import typing
from typing import Any, Callable, TypeVar

from agentic_commons.config import is_enabled
from agentic_commons.logging.core import SUBSYSTEM_COLORS, get_logger

T = TypeVar("T")


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
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            logger = get_logger(subsystem)
            if toggle_path is not None and not is_enabled(toggle_path):
                logger.log_skipped(func.__qualname__, color, toggle_path)
                return _placeholder_for(func)
            visible_args = args[1:] if args and hasattr(args[0], "__dict__") else args
            logger.log_called(func.__qualname__, color, visible_args, kwargs)
            start = time.perf_counter()
            result = func(*args, **kwargs)
            logger.log_returned(
                func.__qualname__, color, result, (time.perf_counter() - start) * 1000
            )
            return result

        return wrapper

    return decorator


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
