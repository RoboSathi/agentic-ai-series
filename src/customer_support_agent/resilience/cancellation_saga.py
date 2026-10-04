"""The cancel-and-refund saga.

Cancelling an unshipped order is one more step in front of a refund:
stop the order, then pay it back. It is not a second payment path — the
refund steps are the `RefundSaga`'s own, reused, so the ledger write and
the conditional issue behave exactly as they do for any refund, and the
refund settles the same way. If paying fails, the order goes back to
pending: the customer keeps an order they paid for rather than losing both
the order and the money.

Both steps are compare-and-swaps — the cancellation on the order state the
caller saw, the refund on the state, total and refundability — so there is
no separate re-read of the order before starting.
"""

from agent_platform.logging import instrument
from agent_platform.primitives.money import Money
from agent_platform.resilience import (
    CompensatingSaga,
    IdempotencyGuard,
    Saga,
    SagaResult,
    SagaStep,
    SagaWorkflow,
)
from customer_support_agent.domain.models import Order, OrderStatus
from customer_support_agent.persistence.repositories import OrdersRepository
from customer_support_agent.resilience.refund_saga import RefundRequest, RefundSaga

_CANCELLED: OrderStatus = "cancelled"
_CANCELLATION_KEY = "cancel:{order_id}:{cents}"


def cancellation_key(order_id: int, amount: Money) -> str:
    """The idempotency key a cancel-and-refund of this order runs under."""
    return _CANCELLATION_KEY.format(order_id=order_id, cents=amount.cents)


@instrument("customer_support_agent")
class CancellationSaga(SagaWorkflow[RefundRequest]):
    """Cancels a pending order and refunds its total, compensably."""

    def __init__(
        self,
        orders: OrdersRepository,
        refund: RefundSaga,
        idempotency: IdempotencyGuard | None = None,
        saga: Saga | None = None,
    ) -> None:
        super().__init__(saga or CompensatingSaga(), idempotency)
        self._orders = orders
        self._refund = refund

    async def cancel(self, order: Order, amount: Money, reason: str) -> SagaResult:
        """Cancel and refund, unwinding whatever committed if a step fails."""
        return await self.execute(RefundRequest(order=order, amount=amount, reason=reason))

    def idempotency_key(self, request: RefundRequest) -> str:
        """Distinct from a plain refund's key, so the two never replay each other."""
        return cancellation_key(request.order.order_id, request.amount)

    def steps(self, request: RefundRequest) -> list[SagaStep]:
        """Cancel first, then the refund saga's own steps."""
        order = request.order
        previous_status = order.order_status

        async def mark_cancelled() -> None:
            if not self._orders.set_order_status(
                order.order_id, _CANCELLED, expected=previous_status
            ):
                raise RuntimeError(
                    f"Order {order.order_id} was no longer '{previous_status}' when the "
                    "cancellation tried to commit; it may have shipped meanwhile"
                )

        async def restore_status() -> None:
            self._orders.set_order_status(order.order_id, previous_status)

        cancel = SagaStep(
            name="mark_order_cancelled", action=mark_cancelled, compensation=restore_status
        )
        return [cancel, *self._refund.steps(request)]
