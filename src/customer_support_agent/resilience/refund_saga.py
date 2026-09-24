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

It also does not trust its caller. It used to: `issue()` took whatever
`Order` it was handed and paid out against it, with no check that the
order was eligible, that the amount was within its total, or that the
snapshot was still current. That was safe only by coincidence — both
call sites happened to check first — and a third caller, or a stale
object held across an await, would have paid out on an order that had
already been refunded. The money-moving step is the wrong place to
assume somebody else was careful, so the preconditions are re-checked
here, against a fresh read, and the state transition is a
compare-and-swap rather than a blind write.
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
from customer_support_agent.domain.models import Order
from customer_support_agent.persistence.repositories import (
    OrdersRepository,
    RefundLedgerRepository,
)

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
        refused = self._refuse(order, amount)
        if refused is not None:
            return refused

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

    def _refuse(self, order: Order, amount: float) -> SagaResult | None:
        """Why this refund must not run, or None if it may.

        Read fresh from the repository rather than taken from the `Order`
        passed in. The caller's copy may have been loaded several awaits
        ago, and the only state worth paying out against is the state as
        it is now.
        """
        current = self._orders.get(order.order_id)
        if current is None:
            return self._refused(f"Order {order.order_id} no longer exists")
        if current.user_id != order.user_id:
            return self._refused(f"Order {order.order_id} does not belong to that customer")
        if current.already_refunded or not current.is_refund_eligible:
            return self._refused(current.refund_ineligibility_reason)
        if amount <= 0:
            return self._refused("A refund must be for more than zero")
        if amount > current.amount:
            return self._refused(
                f"Refund of ${amount:,.2f} exceeds order {order.order_id}'s "
                f"${current.amount:,.2f} total"
            )
        return None

    def _refused(self, reason: str) -> SagaResult:
        """A refund that never started, reported as a failed saga.

        Nothing committed, so there is nothing to compensate — which is
        the distinction between refusing and rolling back.
        """
        self._logger.warning("Refund refused before any step ran", reason=reason)
        return SagaResult(succeeded=False, failed_step="precondition", error=reason)

    def _steps(self, order: Order, amount: float, reason: str, refund_id: str) -> list[SagaStep]:
        """The three steps, each with the action that undoes it."""
        previous_status = order.refund_status

        async def mark_refunded() -> None:
            # Compare-and-swap: the row moves only if it is still in the
            # state the precondition check saw. A concurrent refund then
            # loses here rather than both of them paying out.
            if not self._orders.set_refund_status(
                order.order_id, "completed", expected=previous_status
            ):
                raise RuntimeError(
                    f"Order {order.order_id} was no longer '{previous_status}' when the "
                    "refund tried to commit; another refund may have won the race"
                )

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
