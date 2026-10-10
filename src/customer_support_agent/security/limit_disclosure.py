"""Keeps the refund limit out of what a customer reads.

The model is told the limit so that it holds to it, and told never to say
it. The second instruction is a request like the first: a reply that
names the threshold tells whoever is probing exactly what to aim under.
So the reply is checked here, deterministically, after grounding, and a
reply that states the limit is replaced by the configured decline.

An amount equal to the limit is not a disclosure when a tool showed it
this turn: an order whose total happens to be the limit is quoted as the
record has it.
"""

from collections.abc import Iterable

from agent_platform.hallucination_protection.direct import amounts
from agent_platform.logging import get_logger
from agent_platform.primitives.money import Money
from customer_support_agent.config import get_agent_settings


class LimitDisclosureCheck:
    """Withholds a reply that states the agent's refund limit."""

    def __init__(self, limit: Money | None) -> None:
        self._limit = limit
        self._logger = get_logger("guardrails")

    def withhold(self, reply: str, shown: Iterable[str], order_id: int | None) -> str | None:
        """The decline to send instead of `reply`, or None when the reply may go as written.

        `shown` is the text this turn's record and tools put in front of
        the model: an amount found there was read, not disclosed.
        """
        if self._limit is None or self._limit not in amounts(reply):
            return None
        if any(self._limit in amounts(text) for text in shown):
            return None
        replacement = _decline(order_id)
        self._logger.warning(
            "Reply stated the refund limit; withheld",
            reply=reply,
            replacement=replacement,
            order_id=order_id,
        )
        return replacement


def _decline(order_id: int | None) -> str:
    """The configured decline, naming the order when one is under discussion."""
    settings = get_agent_settings()
    withheld = settings.refund_limit_withheld
    if order_id is None:
        return withheld.without_order.format(contact=settings.support_contact)
    return withheld.with_order.format(order_id=order_id, contact=settings.support_contact)
