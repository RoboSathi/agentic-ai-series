"""Which order the customer is talking about.

A support conversation names an order once and then stops naming it.
"Order 10301?" — "yes, I want a refund" is two turns about one order, and
a component that reads only the current message sees an order in the
first and nothing in the second.

So the reference is resolved once per turn, from three sources in
descending order of directness: what this turn already established, what
the customer typed in this message, and what the customer established
earlier in the session. All three are the customer's own words or a
record derived from them — never something the model proposed, which is
the distinction the Gatekeeper's mandate check depends on.

One resolver rather than one per caller: the read path and the write path
disagreeing about which order is under discussion is how an agent answers
about one order and acts on another.
"""

import re
from collections.abc import Callable, Sequence
from typing import Protocol

from agent_platform.memory.working import WorkingMemory
from customer_support_agent.domain.ports import OrderLookup

# Working memory stores the reference as prose ("order 10301") because a
# person reads that pane. Reading it back is this module's problem.
_STORED_REFERENCE = re.compile(r"\border\s+(\d+)\b", re.IGNORECASE)

# The key under which the established reference travels, in working
# memory and in the turn's metadata alike. One name, so a component
# cannot look under a spelling nobody writes.
ORDER_UNDER_DISCUSSION = "order_under_discussion"

# Whether this turn's intent may change an order (`mode: write`). A refund or
# a cancellation concerns the order under discussion and no other, so its
# context carries that one order or none.
WRITE_INTENT = "write_intent"

# Whether an order is this customer's own: (order id, customer id).
OwnsOrder = Callable[[int, int], bool]

# Whether the words just before an offset in a message refuse what follows: (message, offset).
NegatedAt = Callable[[str, int], bool]


class OrderReferenceRules(Protocol):
    """How a customer's words are read as naming an order (configured by the agent)."""

    @property
    def anchored_patterns(self) -> Sequence[str]: ...

    @property
    def compact_patterns(self) -> Sequence[str]: ...

    @property
    def id_shape(self) -> str: ...


class OrderReferenceResolver:
    """The order under discussion, from the customer and the record.

    What a message names is decided in three steps, most direct first: a
    number tied to the word "order" (or the compact "refund N"); failing
    that, a bare id-shaped number that is one of *this customer's own*
    orders; failing that, nothing. Only when the message names no number
    at all does the order from earlier in the conversation carry over.

    A message that does name a number, none of which is the customer's, is
    not about the old order either. Falling back to it was how "want
    refund for this order -> 10786" refunded, or refused to refund, an
    order the customer had stopped talking about.
    """

    def __init__(
        self,
        rules: OrderReferenceRules,
        owns: OwnsOrder | None = None,
        negated_at: NegatedAt | None = None,
    ) -> None:
        self._named = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in (*rules.anchored_patterns, *rules.compact_patterns)
        ]
        self._shape = re.compile(rules.id_shape)
        self._owns = owns
        self._negated_at = negated_at

    @classmethod
    def from_orders(
        cls, rules: OrderReferenceRules, orders: OrderLookup, negated_at: NegatedAt | None = None
    ) -> "OrderReferenceResolver":
        """A resolver that checks ownership against the customer's own orders."""
        return cls(
            rules,
            lambda order_id, customer_id: (
                orders.get_for_customer(order_id, customer_id) is not None
            ),
            negated_at,
        )

    def resolve(
        self,
        query: str,
        metadata: dict[str, object] | None = None,
        memory: WorkingMemory | None = None,
        customer_id: int | None = None,
    ) -> int | None:
        """The most recent order the customer established, newest first."""
        settled = _as_order_id((metadata or {}).get(ORDER_UNDER_DISCUSSION))
        if settled is not None:
            return settled
        decided, named = self._from_message(query, customer_id)
        return named if decided else self._from_memory(memory)

    def _from_message(self, query: str, customer_id: int | None) -> tuple[bool, int | None]:
        """What this message names: (whether it named anything, and the order if it settles on one).

        An order the customer refused ("do not refund order 10511") is named but is not the
        one under discussion: reading it as such is how the order they said to leave alone
        became the one acted on.
        """
        matches = [m for pattern in self._named for m in pattern.finditer(query)]
        for match in matches:
            if not self._refused(query, match.start()):
                return True, int(match.group("id"))
        if matches:
            return True, None
        return self._from_ownership(query, customer_id)

    def _refused(self, query: str, start: int) -> bool:
        return self._negated_at is not None and self._negated_at(query, start)

    def names_order(self, query: str, customer_id: int | None) -> bool:
        """Whether this message itself asks about an order, never one carried from earlier."""
        return self._from_message(query, customer_id)[1] is not None

    def named(self, query: str, customer_id: int | None) -> set[int]:
        """Every order this message mentions, refused or not."""
        found = {int(m.group("id")) for pattern in self._named for m in pattern.finditer(query)}
        if self._owns is None or customer_id is None:
            return found
        shaped = {int(token) for token in self._shape.findall(query)}
        return found | {order for order in shaped if self._owns(order, customer_id)}

    def _from_ownership(self, query: str, customer_id: int | None) -> tuple[bool, int | None]:
        """A bare id-shaped number, believed only because the record says it is theirs."""
        shaped = {int(token) for token in self._shape.findall(query)}
        if not shaped or self._owns is None or customer_id is None:
            return False, None
        owned = {order_id for order_id in shaped if self._owns(order_id, customer_id)}
        return True, next(iter(owned)) if len(owned) == 1 else None

    def _from_memory(self, memory: WorkingMemory | None) -> int | None:
        """What the customer established on an earlier turn.

        Working memory is recency ordered, so the first match is the most
        recent validated reference. Naming a different order overwrites
        this, which is what makes the carried reference expire.
        """
        if memory is None:
            return None
        value = memory.get(ORDER_UNDER_DISCUSSION)
        if isinstance(value, str):
            match = _STORED_REFERENCE.search(value)
            return int(match.group(1)) if match else None
        return _as_order_id(value)


def _as_order_id(value: object) -> int | None:
    """A positive order id, or nothing. `bool` is not an order id."""
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


__all__ = ["ORDER_UNDER_DISCUSSION", "OrderReferenceResolver", "OwnsOrder"]
