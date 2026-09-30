"""Shared resilience value types (LLD §13)."""

from datetime import datetime
from typing import Any, Awaitable, Callable, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from agentic_commons.observability.types import utc_now

T = TypeVar("T")

# Any async operation the resilience decorators can wrap.
AsyncOperation = Callable[[], Awaitable[Any]]

CircuitState = Literal["closed", "open", "half_open"]


class CircuitBreakerOpenError(Exception):
    """Raised when a breaker refuses a call because the circuit is open.

    Failing fast is the point: a caller queuing behind a dependency that
    is already known to be down converts one outage into two.
    """


class BulkheadFullError(Exception):
    """Raised when a pool has no capacity and the wait budget is exhausted."""


class RateLimitWaitExceededError(Exception):
    """Raised when a rate-limit slot is further away than the wait budget.

    Failing loudly beats hanging: a run stalled behind a long provider
    cooldown should say so rather than look like a slow model.
    """


class NonIdempotentRetryError(Exception):
    """Raised when a retry is attempted without an idempotency key.

    Enforced rather than documented. Retrying a refund without one is how
    a customer is paid twice.
    """


def external_api_error_fields(exc: Exception) -> dict[str, Any] | None:
    """Extract structured error details returned by an external API.

    Supports provider SDK exceptions (for example OpenAI's ``APIStatusError``)
    and raw HTTP exceptions (for example ``httpx.HTTPStatusError``) without
    importing either provider into the resilience layer. Returns ``None`` for
    ordinary local/programming errors so callers only add API fields when an
    external dependency actually failed.
    """
    response = getattr(exc, "response", None)
    status_code = getattr(exc, "status_code", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)

    body = getattr(exc, "body", None)
    if body is None and response is not None:
        try:
            body = response.json()
        except Exception:
            body = None

    if not (status_code is not None or body is not None or response is not None):
        return None

    payload = body if isinstance(body, dict) else {}
    error = payload.get("error")
    nested: dict[str, Any] = error if isinstance(error, dict) else {}

    error_code = (
        nested.get("code")
        or payload.get("code")
        or payload.get("error_code")
        or nested.get("type")
        or payload.get("type")
    )
    reason = (
        nested.get("message")
        or payload.get("message")
        or payload.get("error_message")
        or payload.get("detail")
        or payload.get("title")
    )
    if reason is None and response is not None:
        try:
            text = str(response.text).strip()
        except Exception:
            text = ""
        reason = text or None
    if reason is None:
        reason = str(exc) or type(exc).__name__

    if error_code is None and status_code is not None:
        error_code = f"HTTP_{status_code}"
    if error_code is None:
        error_code = type(exc).__name__.upper()

    return {
        "error_code": str(error_code),
        "reason": str(reason),
        "http_status": status_code,
    }


# Error codes for a dependency that failed before it could answer at all.
TIMEOUT_ERROR_CODE = "TIMEOUT"
CONNECTION_ERROR_CODE = "CONNECTION_ERROR"
# Transport exceptions from HTTP clients the resilience layer does not import,
# recognised by class name anywhere in their hierarchy.
_TIMEOUT_CLASS_MARKERS = ("Timeout",)
_CONNECTION_CLASS_MARKERS = ("ConnectError", "ConnectionError")


def dependency_failure_fields(exc: Exception) -> dict[str, Any]:
    """Structured fields for any failed call to an external dependency.

    `external_api_error_fields` only answers when the dependency replied.
    A timeout never gets a reply, and `str(TimeoutError())` is empty — so
    an outage used to be logged with a blank code, a blank reason and no
    status, indistinguishable from a bug. This always names the failure.
    """
    exception_type = type(exc).__name__
    replied = external_api_error_fields(exc)
    if replied is not None:
        return {**replied, "exception_type": exception_type}
    code = _transport_error_code(exc)
    return {
        "error_code": code,
        "reason": str(exc) or _TRANSPORT_REASONS.get(code, exception_type),
        "http_status": None,
        "exception_type": exception_type,
    }


_TRANSPORT_REASONS = {
    TIMEOUT_ERROR_CODE: "no response before the per-attempt deadline",
    CONNECTION_ERROR_CODE: "could not connect to the endpoint",
}


def _transport_error_code(exc: Exception) -> str:
    """Classify a reply-less failure as a timeout, a connection error, or by type."""
    names = [cls.__name__ for cls in type(exc).__mro__]
    if isinstance(exc, TimeoutError) or _any_marker(names, _TIMEOUT_CLASS_MARKERS):
        return TIMEOUT_ERROR_CODE
    if isinstance(exc, ConnectionError) or _any_marker(names, _CONNECTION_CLASS_MARKERS):
        return CONNECTION_ERROR_CODE
    return type(exc).__name__.upper()


def _any_marker(names: list[str], markers: tuple[str, ...]) -> bool:
    return any(marker in name for name in names for marker in markers)


class Event(BaseModel):
    """One published message."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    topic: str
    payload: dict[str, Any] = Field(default_factory=dict)
    published_at: datetime = Field(default_factory=utc_now)
    trace_id: str = ""


class SagaStep(BaseModel):
    """An action paired with the action that undoes it (Command pattern).

    Bundling them as one value is what makes compensation possible: a
    step that reaches the ledger without its inverse cannot be rolled
    back later, whatever the orchestrator intends.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    action: AsyncOperation
    compensation: AsyncOperation | None = None


class SagaResult(BaseModel):
    """Outcome of a saga, including what had to be undone."""

    succeeded: bool
    completed: list[str] = Field(default_factory=list)
    compensated: list[str] = Field(default_factory=list)
    failed_step: str | None = None
    error: str | None = None
    compensation_errors: dict[str, str] = Field(default_factory=dict)
