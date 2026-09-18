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

from contextvars import ContextVar

from pydantic import BaseModel

from agentic_commons.logging import get_logger

_logger = get_logger("customer_support_agent")


class Customer(BaseModel):
    """The authenticated customer a request is being handled for."""

    customer_id: str
    email: str = ""
    verified: bool = True


_CURRENT_CUSTOMER: ContextVar[Customer | None] = ContextVar("current_customer", default=None)


def set_current_customer(customer: Customer | None) -> None:
    """Bind the customer this request is for."""
    _CURRENT_CUSTOMER.set(customer)


def current_customer() -> Customer | None:
    """The customer this request is for, or None if unauthenticated."""
    return _CURRENT_CUSTOMER.get()


def current_customer_id() -> str:
    """The authenticated customer's id, or empty when unauthenticated.

    Callers must treat empty as "scope to nothing", never as "scope to
    everything" — an unauthenticated request should see no orders, not
    every order.
    """
    customer = current_customer()
    if customer is None:
        _logger.warning("No authenticated customer bound to this request")
        return ""
    return customer.customer_id
