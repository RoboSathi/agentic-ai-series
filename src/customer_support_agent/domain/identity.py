"""Who the harness is acting for (LLD §21).

The authenticated customer is held in a context variable, set by the
harness at the start of a request and read by the tools when they scope a
query.

It is deliberately *not* a tool argument. Anything in a tool's schema is
something the model fills in, and a model that can name the customer can
name a different one — the model would be choosing whose data it reads.
Identity comes from authentication, upstream of anything the model says,
so the only component that can set it is the one that authenticated it.

This mirrors `logging.trace_context`: ambient request state that flows
through every layer without being threaded through every signature.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from pydantic import BaseModel

from agent_platform.logging import get_logger

_logger = get_logger("customer_support_agent")


class Customer(BaseModel):
    """The authenticated customer a request is being handled for."""

    customer_id: int
    name: str = ""
    email: str = ""
    verified: bool = True


_CURRENT_CUSTOMER: ContextVar[Customer | None] = ContextVar("current_customer", default=None)


def set_current_customer(customer: Customer | None) -> None:
    """Bind the customer this request is for."""
    _CURRENT_CUSTOMER.set(customer)


@contextmanager
def bound_customer(customer: Customer | None) -> Iterator[None]:
    """Bind the customer for one request, and restore what was bound before on exit.

    A bare `set_current_customer` in the caller's own context outlives the
    request: a review approved from the console bound that customer into the
    console's context, and the next session's first turn found them already
    "verified" without a code. Bound here, a request's identity ends with it.
    """
    token = _CURRENT_CUSTOMER.set(customer)
    try:
        yield
    finally:
        _CURRENT_CUSTOMER.reset(token)


def current_customer() -> Customer | None:
    """The customer this request is for, or None if unauthenticated."""
    return _CURRENT_CUSTOMER.get()


def current_customer_id() -> int | None:
    """The authenticated customer's id, or None when unauthenticated.

    Callers must treat None as "scope to nothing", never as "scope to
    everything" — an unauthenticated request should see no orders, not
    every order.
    """
    customer = current_customer()
    if customer is None:
        _logger.warning("No authenticated customer bound to this request")
        return None
    return customer.customer_id
