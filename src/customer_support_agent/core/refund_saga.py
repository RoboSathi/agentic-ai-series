"""The refund saga (E35, LLD §13/§21).

Issuing a refund touches three things that cannot share a transaction:
the order's refund state, the refund ledger, and the customer
notification. A crash between them leaves money moved and the order
still marked unrefunded — or worse, the reverse.

So each step carries its own compensation and the saga unwinds in
reverse on failure. Notification is deliberately last: it is the one
step that cannot be compensated, because you cannot un-tell a customer
they were refunded. Steps that cannot be undone go last, after
everything that can.
"""

import uuid
from typing import Awaitable, Callable

from agentic_commons.logging import get_logger, instrument
from agentic_commons.resilience import (
    CompensatingSaga,
    SagaResult,
    SagaStep,
    SqliteIdempotencyGuard,
)
from customer_support_agent.core.repositories import (
    OrdersRepository,
    RefundLedgerRepository,
)
from customer_support_agent.core.schema import Order

# Notifier is injected: the agent does not own how a customer is reached.
Notifier = Callable[[int, str], Awaitable[None]]


@instrument("customer_support_agent")
class RefundSaga:
    """Issues a refund as a compensable sequence of steps."""

    def __init__(
        self,
        orders: OrdersRepository,
        ledger: RefundLedgerRepository,
        notifier: Notifier | None = None,
        idempotency: SqliteIdempotencyGuard | None = None,
    ) -> None:
        self._orders = orders
        self._ledger = ledger
        self._notifier = notifier
        self._idempotency = idempotency
        self._saga = CompensatingSaga()
        self._logger = get_logger("customer_support_agent")

    async def issue(self, order: Order, amount: float, reason: str) -> SagaResult:
        """Run the refund, compensating whatever committed if a step fails.

        The idempotency key is the order plus the amount: a retried
        refund request for the same order and amount replays the first
        outcome instead of paying twice.
        """
        refund_id = str(uuid.uuid4())
        key = f"refund:{order.order_id}:{amount:.2f}"

        async def run() -> SagaResult:
            return await self._saga.run(self._steps(order, amount, reason, refund_id))

        if self._idempotency is None:
            return await run()
        recorded = self._idempotency.recorded_result(key)
        if recorded is not None:
            self._logger.info("Replaying prior refund outcome", key=key)
            return SagaResult.model_validate(recorded)
        result = await run()
        if result.succeeded:
            self._idempotency.remember(key, result.model_dump(mode="json"))
        return result

    def _steps(self, order: Order, amount: float, reason: str, refund_id: str) -> list[SagaStep]:
        """The three steps, each with the action that undoes it."""
        previous_status = order.refund_status

        async def mark_refunded() -> None:
            if not self._orders.set_refund_status(order.order_id, "completed"):
                raise RuntimeError(f"Order {order.order_id} not found")

        async def restore_status() -> None:
            self._orders.set_refund_status(order.order_id, previous_status)

        async def record_payment() -> None:
            self._ledger.record(refund_id, order.order_id, order.user_id, amount, reason)

        async def reverse_payment() -> None:
            self._ledger.reverse(refund_id)

        async def notify() -> None:
            if self._notifier is not None:
                await self._notifier(
                    order.user_id,
                    f"Your refund of ${amount:.2f} for order {order.order_id} is on its way.",
                )

        return [
            SagaStep(name="mark_order_refunded", action=mark_refunded, compensation=restore_status),
            SagaStep(name="record_payment", action=record_payment, compensation=reverse_payment),
            # No compensation: a sent notification cannot be unsent, which
            # is exactly why this step runs only after the rest committed.
            SagaStep(name="notify_customer", action=notify, compensation=None),
        ]
