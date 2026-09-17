"""Call-logging decorator for Phase 0 stub methods."""

import functools
import inspect
import time
from typing import Any, Callable, TypeVar

from agentic_commons.config import get_settings
from agentic_commons.logging.core import get_logger

T = TypeVar("T")

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
    "tools": "white",
    "gatekeeper": "bright_red",
    "hallucination_protection": "orange3",
    "customer_support_agent": "bright_green",
}


def log_call(
    subsystem: str, toggle_path: str | None = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator factory for logging stub method calls.

    Args:
        subsystem: The subsystem name (used for color).
        toggle_path: Dotted path to the toggle in Settings (e.g.,
            "security.pii.enabled"). Defaults to "{subsystem}.enabled".

    Returns:
        A decorator that wraps the function with call logging, toggle
        checking, and timing.
    """
    color = SUBSYSTEM_COLORS.get(subsystem, "white")
    path = toggle_path or f"{subsystem}.enabled"

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            logger = get_logger(subsystem)
            if not _resolve_toggle(path):
                logger.log_skipped(func.__qualname__, color, path)
                return _placeholder_for(func)
            start = time.perf_counter()
            # Filter out 'self' from args for cleaner logging
            filtered_args = args[1:] if args and hasattr(args[0], '__dict__') else args
            logger.log_called(func.__qualname__, color, filtered_args, kwargs)
            result = func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.log_returned(func.__qualname__, color, result, elapsed_ms)
            return result

        return wrapper

    return decorator


def _resolve_toggle(path: str) -> bool:
    """Resolve a dotted path to a toggle value in Settings."""
    settings = get_settings()
    node: Any = settings
    for part in path.split("."):
        node = getattr(node, part, None)
        if node is None:
            return False
    return bool(node)


def _placeholder_for(func: Callable[..., T]) -> T:
    """Return a type-appropriate dummy value based on func's return annotation.

    Returns a zero-arg or all-default constructed instance for dataclass/
    BaseModel return types, or built-in defaults for bool/str/dict/list/None.

    Raises NotImplementedError for unhandled return annotations.
    """
    sig = inspect.signature(func)
    return_annotation = sig.return_annotation

    if return_annotation is inspect.Signature.empty or return_annotation is None:
        return None

    if return_annotation is bool:
        return False
    if return_annotation is str:
        return ""
    if return_annotation is int:
        return 0
    if return_annotation is float:
        return 0.0
    if return_annotation is dict or return_annotation is dict[Any, Any]:
        return {}
    if return_annotation is list or return_annotation is list[Any]:
        return []
    if return_annotation is tuple or return_annotation is tuple[Any, ...]:
        return ()
    if return_annotation is set or return_annotation is set[Any]:
        return set()

    try:
        if hasattr(return_annotation, "__origin__"):
            origin = return_annotation.__origin__
            if origin is dict:
                return {}
            if origin is list:
                return []
            if origin is tuple:
                return ()
            if origin is set:
                return set()

        if hasattr(return_annotation, "model_validate"):
            return return_annotation.model_construct()

        if hasattr(return_annotation, "__dataclass_fields__"):
            return return_annotation()

        return return_annotation()
    except Exception:
        raise NotImplementedError(
            f"Cannot create placeholder for return type {return_annotation} "
            f"in {func.__qualname__}"
        )
