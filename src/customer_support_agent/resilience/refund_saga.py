"""The refund saga (E35, LLD §13/§21).

Issuing a refund touches two things that cannot share a transaction: the
order's refund state and the refund ledger. A crash between them leaves
money moved and the order still marked unrefunded — or worse, the reverse.
So each step carries its own compensation and the saga unwinds in reverse
on failure.

A refund is *issued* here, not completed. Like a card payment it settles
later (`RefundSettlement`), and until it settles it can be voided. That is
why the customer notification is not a step of this saga any more: it is
the one thing that cannot be undone, so it waits for settlement, after
which nothing is undone either.

It does not trust its caller. Every fact the refund depends on — the
refund state the caller saw, the order's full total, a refundable order
state — is part of one conditional update. It used to re-read the order
and re-run the refund policy before starting; the conditional write makes
that read redundant, and unlike a read it cannot go stale before the write.

The skeleton — refuse, replay, run, remember — is the platform's
`SagaWorkflow`. This class supplies only what is the refund's own: the
steps and the key that recognises a retried refund.
"""

import uuid
from datetime import timedelta

from pydantic import BaseModel, ConfigDict

from agent_platform.core.primitives.money import Money
from agent_platform.logging import instrument
from agent_platform.observability.types import utc_now
from agent_platform.resilience import (
    CompensatingSaga,
    IdempotencyGuard,
    Saga,
    SagaResult,
    SagaStep,
    SagaWorkflow,
)
from agent_platform.runtime.executor import current_execution_key
from customer_support_agent.config import get_agent_settings
from customer_support_agent.domain.models import Order
from customer_support_agent.persistence.repositories import (
    OrdersRepository,
    RefundLedgerRepository,
)

_REFUND_KEY = "refund:{order_id}:{cents}"


class RefundRequest(BaseModel):
    """One refund to issue: which order, how much, and why."""

    model_config = ConfigDict(frozen=True)

    order: Order
    amount: Money
    reason: str


def refund_key(order_id: int, amount: Money) -> str:
    """The idempotency key a refund of this order, for this amount, runs under."""
    return _REFUND_KEY.format(order_id=order_id, cents=amount.cents)


@instrument("customer_support_agent")
class RefundSaga(SagaWorkflow[RefundRequest]):
    """Issues a refund as a compensable sequence of steps."""

    def __init__(
        self,
        orders: OrdersRepository,
        ledger: RefundLedgerRepository,
        idempotency: IdempotencyGuard | None = None,
        saga: Saga | None = None,
        window_seconds: int | None = None,
    ) -> None:
        super().__init__(saga or CompensatingSaga(), idempotency)
        self._orders = orders
        self._ledger = ledger
        self._window = timedelta(
            seconds=window_seconds
            if window_seconds is not None
            else get_agent_settings().settlement.window_seconds
        )

    async def issue(self, order: Order, amount: Money, reason: str) -> SagaResult:
        """Issue the refund, compensating whatever committed if a step fails."""
        return await self.execute(RefundRequest(order=order, amount=amount, reason=reason))

    def idempotency_key(self, request: RefundRequest) -> str:
        """The order plus the amount: a retried refund replays, never pays twice."""
        return refund_key(request.order.order_id, request.amount)

    def steps(self, request: RefundRequest) -> list[SagaStep]:
        """Mark it issued, then record the payment — each with the action that undoes it."""
        order, amount, reason = request.order, request.amount, request.reason
        previous_status = order.refund_status
        refund_id = str(uuid.uuid4())

        async def mark_issued() -> None:
            settles_at = (utc_now() + self._window).isoformat()
            if not self._orders.issue_refund(order.order_id, previous_status, amount, settles_at):
                raise RuntimeError(
                    f"Order {order.order_id} could not be refunded {amount}: it is no longer "
                    f"'{previous_status}', refundable, or totalling {amount} — another change "
                    "may have won the race"
                )

        async def restore_status() -> None:
            self._orders.restore_refund(order.order_id, previous_status)

        async def record_payment() -> None:
            # The Executor's key for this call, kept with the payment: a void
            # releases it, so the same refund asked for again is paid again.
            self._ledger.record(
                refund_id, order.order_id, order.user_id, amount, reason, current_execution_key()
            )

        async def reverse_payment() -> None:
            self._ledger.reverse(refund_id)

        return [
            SagaStep(name="mark_refund_issued", action=mark_issued, compensation=restore_status),
            SagaStep(name="record_payment", action=record_payment, compensation=reverse_payment),
        ]
