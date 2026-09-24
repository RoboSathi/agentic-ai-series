"""What this agent's write actions claim, and how to check them.

`agentic_commons.verification` owns the discipline — read the thing back,
compare it, refuse to call an unverified result authoritative. It owns
none of the vocabulary: it does not know that a refund is recorded in two
places, or that a queued review leaves an order `pending_review`.

That is what this module supplies. `intended` says what the world should
look like after a call; `observed` goes and looks.
"""

from typing import Any

from agentic_commons.tools.types import ToolCall, ToolResult
from agentic_commons.verification import IntendedEffect
from customer_support_agent.persistence import (
    HitlQueueRepository,
    OrdersRepository,
    RefundLedgerRepository,
)

# The two actions that change something, and what each leaves behind.
PROCESS_REFUND = "process_refund"
QUEUE_FOR_APPROVAL = "queue_for_approval"


class RefundEffectSource:
    """What a refund or an escalation was meant to do, and what it did."""

    def __init__(
        self,
        orders: OrdersRepository,
        refunds: RefundLedgerRepository,
        hitl: HitlQueueRepository,
    ) -> None:
        self._orders = orders
        self._refunds = refunds
        self._hitl = hitl

    def intended(self, call: ToolCall, result: ToolResult) -> IntendedEffect | None:
        """The change this call claimed to make, or None if it claims none.

        A failed call claims nothing: there is an error to report, not an
        effect to verify. A read claims nothing either.
        """
        if not result.success:
            return None
        action = _bare(call.action)
        order_id = result.data.get("order_id") or call.params.get("order_id")
        if order_id is None:
            return None
        if action == PROCESS_REFUND:
            return IntendedEffect(
                action=action,
                target=f"order:{order_id}",
                expected={
                    "refund_status": "completed",
                    # The money, not the status. An order marked refunded
                    # with nothing in the ledger is the failure this check
                    # exists to catch.
                    "settled_amount": float(result.data.get("amount", 0.0)),
                },
            )
        if action == QUEUE_FOR_APPROVAL:
            return IntendedEffect(
                action=action,
                target=f"order:{order_id}",
                expected={"refund_status": "pending_review", "review_pending": True},
            )
        return None

    def observed(self, effect: IntendedEffect) -> dict[str, Any]:
        """What the system of record says about that order, right now."""
        order_id = int(effect.target.split(":", maxsplit=1)[1])
        order = self._orders.get(order_id)
        if order is None:
            return {}
        return {
            "refund_status": order.refund_status,
            "settled_amount": self._refunds.settled_for_order(order_id),
            "review_pending": self._hitl.pending_for_order(order_id) is not None,
        }


def _bare(action: str) -> str:
    """The tool's own name, without its namespace."""
    return action.rsplit(".", maxsplit=1)[-1]
