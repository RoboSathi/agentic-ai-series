"""What this customer actually asked for, this turn.

`agentic_commons` owns the shape of a mandate and the check that a write
is covered by one. It does not own what counts as evidence that something
was asked for — that is this agent's vocabulary, and it lives here.

The rule this module exists to hold: a mandate is built from the customer
and the record, never from the model. An order id the customer typed, or
chose from a menu they were shown; an amount read from the authoritative
order. Never a number the model proposed — a mandate derived from the
model's own suggestion authorizes the model to authorize itself, which is
the thing mandates exist to prevent.
"""

import re

from agentic_commons.gatekeeper.mandate import ActionMandate
from agentic_commons.logging import get_logger, instrument
from customer_support_agent.config import get_agent_settings
from customer_support_agent.persistence import OrdersRepository

# The write actions a customer can ask for, and what "asking" looks like.
# Both act on one order and are bounded by that order's total.
_WRITE_ACTIONS = ("process_refund", "queue_for_approval")


@instrument("customer_support_agent")
class RefundMandateIssuer:
    """Issues a refund mandate when a customer asked for one, and not otherwise."""

    def __init__(self, orders: OrdersRepository) -> None:
        self._orders = orders
        settings = get_agent_settings()
        # The same phrases the intent router matches on. One list, so a
        # phrase added there cannot quietly fail to grant a mandate here.
        self._phrases = [
            phrase.casefold()
            for intent in settings.intents
            if intent.mode == "write"
            for phrase in intent.phrases
        ]
        self._reference = re.compile(settings.followup.reference_pattern)
        self._logger = get_logger("customer_support_agent")

    def mandates_for(self, subject_id: str, query: str, turn: int) -> list[ActionMandate]:
        """Every write this turn's request authorizes.

        Returns nothing when the customer did not ask for a write, which
        is the common case and the point: a turn that asked where a parcel
        is authorizes no refund, however the conversation later drifts.
        """
        if not self._asked_for_a_write(query):
            return []
        order_id = self._order_named(query)
        if order_id is None:
            # They asked for a refund but named no order. A mandate needs
            # a target, so the agent has to establish which order first.
            self._logger.info("A write was requested with no order named", turn=turn)
            return []
        order = self._orders.get_for_customer(order_id, int(subject_id))
        if order is None:
            return []
        # Bounded by the authoritative total, not by what anyone claimed.
        return [
            ActionMandate(
                subject_id=subject_id,
                action=action,
                target=f"order:{order_id}",
                max_amount=order.amount,
                granted_by="user_task",
                turn=turn,
                evidence=f"the customer asked about order {order_id} in this turn",
            )
            for action in _WRITE_ACTIONS
        ]

    def _asked_for_a_write(self, query: str) -> bool:
        """Whether this message asks for something that moves money."""
        lowered = query.casefold()
        return any(phrase in lowered for phrase in self._phrases)

    def _order_named(self, query: str) -> int | None:
        """The order this message names, if it names exactly one.

        Two different order numbers in one sentence is ambiguity, not a
        mandate for both. Refusing is the safe reading.
        """
        found = {int(match) for match in self._reference.findall(query)}
        return found.pop() if len(found) == 1 else None
