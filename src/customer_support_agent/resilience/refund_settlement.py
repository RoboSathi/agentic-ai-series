"""Refund settlement: an issued refund is voidable until it settles.

Card payments work this way, and so does this agent. A refund is *issued*
first: the payment is recorded and the order marked, and the Gatekeeper
calls the action VOIDABLE. Until the settlement window closes a person may
void it — the payment is reversed and the order goes back to no refund.
Once it settles it is *completed*: the customer is told the money is on
its way, and from then on the only remedy is another refund.

Void and settle are each one compare-and-swap on `issued`, so whichever
reaches an order first is the one that happened; the other finds nothing
left to do and says so.

A void also releases the idempotency records the refund was made under —
the refund saga's and the Executor's. A remembered result means "this was
already done", which a void has just made false: without the release, the
same refund asked for again would replay the voided one instead of paying.
"""

from collections.abc import Awaitable, Callable
from typing import Literal

from pydantic import BaseModel

from agent_platform.audit import AuditEntry, AuditLedger, NullAuditLedger
from agent_platform.core.primitives.money import Money
from agent_platform.logging import get_logger, instrument, logged
from agent_platform.logging.trace_context import get_current_trace_id
from agent_platform.observability.types import utc_now
from agent_platform.resilience import IdempotencyGuard
from customer_support_agent.domain.models import Order
from customer_support_agent.persistence.repositories import (
    OrdersRepository,
    RefundLedgerRepository,
)
from customer_support_agent.resilience.cancellation_saga import cancellation_key
from customer_support_agent.resilience.refund_saga import refund_key

# Notifier is injected: the agent does not own how a customer is reached.
Notifier = Callable[[int, str], Awaitable[None]]
SettlementState = Literal["voided", "settled", "not_issued", "already_settled"]

_ISSUED = "issued"
_COMPLETED = "completed"
_NONE = "none"
_SETTLED_NOTICE = "Your refund of {amount} for order {order_id} has settled and is on its way."


class SettlementOutcome(BaseModel):
    """What one void or settle did to one order."""

    order_id: int
    state: SettlementState
    amount: Money | None = None


@instrument("customer_support_agent")
class RefundSettlement:
    """Voids issued refunds before they settle, and settles them when their window closes."""

    def __init__(
        self,
        orders: OrdersRepository,
        ledger: RefundLedgerRepository,
        idempotency: IdempotencyGuard | None = None,
        audit: AuditLedger | None = None,
        notifier: Notifier | None = None,
    ) -> None:
        self._orders = orders
        self._ledger = ledger
        self._idempotency = idempotency
        self._audit = audit or NullAuditLedger()
        self._notifier = notifier
        self._logger = get_logger("customer_support_agent")

    def issued(self, user_id: int | None = None) -> list[Order]:
        """Refunds still voidable, soonest to settle first; one customer's when given."""
        return self._orders.issued_refunds(user_id)

    @logged("customer_support_agent")
    def void(self, order_id: int, actor: str) -> SettlementOutcome:
        """Reverse an issued refund before it settles, and release its idempotency records."""
        order = self._orders.fresh(order_id)
        refused = self._not_voidable(order_id, order)
        if refused is not None or order is None:
            return refused or SettlementOutcome(order_id=order_id, state="not_issued")
        if not self._orders.close_refund(order_id, _NONE):
            # Settled (or voided) between the read and the swap: report what won.
            return self._not_voidable(order_id, self._orders.fresh(order_id)) or SettlementOutcome(
                order_id=order_id, state="not_issued"
            )
        released = self._ledger.reverse_for_order(order_id)
        self._release(order, released)
        self._record(order, actor, "void_refund", "voided before settlement")
        return SettlementOutcome(order_id=order_id, state="voided", amount=order.amount)

    @logged("customer_support_agent")
    async def settle(self, order_id: int, actor: str) -> SettlementOutcome:
        """Close one refund's window now: it completes, and the customer is told."""
        order = self._orders.fresh(order_id)
        if order is None or not self._orders.close_refund(order_id, _COMPLETED):
            return SettlementOutcome(order_id=order_id, state="not_issued")
        # Last, and only now: a notification cannot be unsent, so it waits
        # for the one transition nothing undoes.
        if self._notifier is not None:
            await self._notifier(
                order.user_id,
                _SETTLED_NOTICE.format(amount=order.amount, order_id=order_id),
            )
        self._record(order, actor, "settle_refund", "settled")
        return SettlementOutcome(order_id=order_id, state="settled", amount=order.amount)

    async def settle_due(self, actor: str) -> list[SettlementOutcome]:
        """Settle the refunds whose window has closed."""
        due = self._orders.refunds_due(utc_now().isoformat())
        return [await self.settle(order.order_id, actor) for order in due]

    @staticmethod
    def _not_voidable(order_id: int, order: Order | None) -> SettlementOutcome | None:
        """Why this order's refund cannot be voided, or None if it can."""
        if order is not None and order.refund_status == _COMPLETED:
            return SettlementOutcome(
                order_id=order_id, state="already_settled", amount=order.amount
            )
        if order is None or order.refund_status != _ISSUED:
            return SettlementOutcome(order_id=order_id, state="not_issued")
        return None

    def _release(self, order: Order, execution_keys: list[str]) -> None:
        """Forget every idempotency record the voided refund was made under."""
        if self._idempotency is None:
            return
        keys = [
            refund_key(order.order_id, order.amount),
            cancellation_key(order.order_id, order.amount),
            *execution_keys,
        ]
        for key in keys:
            self._idempotency.forget(key)
        self._logger.info("Released idempotency records of a voided refund", keys=keys)

    def _record(self, order: Order, actor: str, action: str, rationale: str) -> None:
        """Audit a settlement transition under the person (or sweep) that made it."""
        self._audit.record(
            AuditEntry(
                trace_id=get_current_trace_id() or f"settlement-{order.order_id}",
                actor=actor,
                action=action,
                authority_context={
                    "target": f"order:{order.order_id}",
                    "amount": str(order.amount),
                    "from": _ISSUED,
                },
                decision="allow",
                rationale=rationale,
            )
        )
