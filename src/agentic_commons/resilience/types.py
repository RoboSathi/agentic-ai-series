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


class NonIdempotentRetryError(Exception):
    """Raised when a retry is attempted without an idempotency key.

    Enforced rather than documented. Retrying a refund without one is how
    a customer is paid twice.
    """


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
