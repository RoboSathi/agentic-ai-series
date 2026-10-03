"""Which order a write may act on.

`agent_platform` owns the shape of a scope and the check that a write is
covered by one. It does not own what counts as an established target —
that is this agent's vocabulary, and it lives here.

This module used to answer a different question: *did the customer ask
for a refund?* It answered it by testing the message against a list of
phrases, and it was wrong in both directions. One typo — "redunf for
order 10979" — matched nothing and denied a legitimate refund. Meanwhile
the bare phrase "refund" matched "do not refund order 10979" and "i do
not want a refund", so a customer *declining* a refund authorised one.

Both faults were the same mistake: a keyword list standing in for
language understanding, inside the authorization path. Reading a sentence
is the model's job, and the model is good at it. The Gatekeeper's job is
the deterministic question the model cannot be trusted with — *may this
refund be made?* — and that lives in `business_rules.py`, which checks
ownership, eligibility, prior refunds, the amount against the
authoritative total, and the agent's limit.

What survives here is the one part that never needed language at all: a
write must act on the order **this conversation established**. That is an
integer the customer typed or picked from a menu, resolved before the
model ran. Without it a mis-read could refund a different order the
customer happens to own — ownership and eligibility would both pass, and
the wrong money would move. With it, the target is always one they named.
"""

from collections.abc import Mapping

from agent_platform.gatekeeper.mandate import ActionMandate
from agent_platform.logging import get_logger, instrument
from customer_support_agent.domain.order_reference import ORDER_UNDER_DISCUSSION
from customer_support_agent.persistence import OrdersRepository

# The write actions that act on one order and are bounded by its total.
_WRITE_ACTIONS = ("process_refund", "queue_for_approval", "cancel_order")


@instrument("customer_support_agent")
class EstablishedOrderScope:
    """Scopes writes to the order the customer established, and no other."""

    def __init__(self, orders: OrdersRepository) -> None:
        self._orders = orders
        self._logger = get_logger("customer_support_agent")

    def mandates_for(
        self,
        subject_id: str,
        query: str,
        turn: int,
        context: Mapping[str, object] | None = None,
    ) -> list[ActionMandate]:
        """Every write the established order permits, bounded by its total.

        `query` is deliberately unused. Nothing in this path reads
        customer prose: interpreting what they meant is the model's work,
        and a keyword list doing it badly is worse than not doing it.
        """
        order_id = self._established(context)
        if order_id is None:
            # No order has been established, this turn or earlier, so
            # there is no target a write could legitimately act on.
            self._logger.info("A write was proposed with no order established", turn=turn)
            return []
        # Re-read under the customer's own id. The reference may have been
        # carried from an earlier turn, but ownership and the amount are
        # never carried — they are read again, now.
        order = self._orders.get_for_customer(order_id, int(subject_id))
        if order is None:
            return []
        return [
            ActionMandate(
                subject_id=subject_id,
                action=action,
                target=f"order:{order_id}",
                # Bounded by the authoritative total, not by what anyone claimed.
                max_amount=order.amount,
                granted_by="user_task",
                turn=turn,
                evidence=f"the customer established order {order_id} in this conversation",
            )
            for action in _WRITE_ACTIONS
        ]

    def _established(self, context: Mapping[str, object] | None) -> int | None:
        """The order this conversation settled on, before the model ran.

        Read from the turn's own metadata, which `choose_order` resolves
        from what the customer typed or picked. Deliberately not the
        proposed call's `order_id`: that arrives later, from the model,
        and reading it here would let a proposal supply its own target.
        """
        value = (context or {}).get(ORDER_UNDER_DISCUSSION)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            return None
        return value
