"""What the domain needs from storage, stated by the domain (dependency inversion).

The refund and cancellation rules need to read an order, scoped to the
customer asking. They do not need to know that orders live in SQLite, or
that a repository exists at all: they depend on this Protocol, and the
persistence layer supplies something that satisfies it. The arrow points
from `persistence` to `domain`, never back.
"""

from typing import Protocol

from customer_support_agent.domain.models import Order


class OrderLookup(Protocol):
    """Reads one order, but only if the given customer owns it."""

    def get_for_customer(self, order_id: int, customer_id: int | None) -> Order | None: ...


__all__ = ["OrderLookup"]
