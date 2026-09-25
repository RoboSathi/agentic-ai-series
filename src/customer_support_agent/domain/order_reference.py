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

from agentic_commons.memory.working import WorkingMemory

# An order id only counts when the customer anchored it to the word
# "order". A bare four-digit number is a menu choice, a quantity or a
# year far more often than it is an order, and guessing wrong here
# points every downstream check at the wrong record.
_SPOKEN_REFERENCE = re.compile(
    r"\border\s*(?:#|no\.?|number)?\s*(?P<a>\d{4,})\b|\b(?P<b>\d{4,})\s+order\b",
    re.IGNORECASE,
)

# Working memory stores the reference as prose ("order 10301") because a
# person reads that pane. Reading it back is this module's problem.
_STORED_REFERENCE = re.compile(r"\border\s+(\d+)\b", re.IGNORECASE)

# The key under which the established reference travels, in working
# memory and in the turn's metadata alike. One name, so a component
# cannot look under a spelling nobody writes.
ORDER_UNDER_DISCUSSION = "order_under_discussion"


class OrderReferenceResolver:
    """The order under discussion, from the customer and the record."""

    def resolve(
        self,
        query: str,
        metadata: dict[str, object] | None = None,
        memory: WorkingMemory | None = None,
    ) -> int | None:
        """The most recent order the customer established, newest first."""
        return (
            self._from_metadata(metadata)
            or extract_order_id(query)
            or self._from_memory(memory)
        )

    def _from_metadata(self, metadata: dict[str, object] | None) -> int | None:
        """What an earlier stage of this same turn already settled."""
        return _as_order_id((metadata or {}).get(ORDER_UNDER_DISCUSSION))

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


def extract_order_id(message: str) -> int | None:
    """The order id a message names, only when anchored to "order"."""
    match = _SPOKEN_REFERENCE.search(message)
    if match is None:
        return None
    raw = match.group("a") or match.group("b")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _as_order_id(value: object) -> int | None:
    """A positive order id, or nothing. `bool` is not an order id."""
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


__all__ = ["ORDER_UNDER_DISCUSSION", "OrderReferenceResolver", "extract_order_id"]
