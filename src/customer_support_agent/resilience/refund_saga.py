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

The skeleton — refuse, replay, run, remember — is the platform's
`SagaWorkflow`. This class supplies only what is the refund's own: the
three steps, the key that recognises a retried refund, and the policy
re-check that refuses to start one.
"""

import uuid
from typing import Awaitable, Callable

from pydantic import BaseModel, ConfigDict

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
from customer_support_agent.domain.models import Order
from customer_support_agent.domain.refund_policy import RefundPolicy
from customer_support_agent.persistence.repositories import (
    OrdersRepository,
    RefundLedgerRepository,
)

# Notifier is injected: the agent does not own how a customer is reached.
Notifier = Callable[[int, str], Awaitable[None]]


class RefundRequest(BaseModel):
    """One refund to issue: which order, how much, and why."""

    model_config = ConfigDict(frozen=True)

    order: Order
    amount: Money
    reason: str


@instrument("customer_support_agent")
class RefundSaga(SagaWorkflow[RefundRequest]):
    """Issues a refund as a compensable sequence of steps."""

    def __init__(
        self,
        orders: OrdersRepository,
        ledger: RefundLedgerRepository,
        notifier: Notifier | None = None,
        idempotency: IdempotencyGuard | None = None,
        refund_policy: RefundPolicy | None = None,
        saga: Saga | None = None,
    ) -> None:
        super().__init__(saga or CompensatingSaga(), idempotency)
        self._orders = orders
        self._refund_policy = refund_policy or RefundPolicy(orders)
        self._ledger = ledger
        self._notifier = notifier

    async def issue(self, order: Order, amount: Money, reason: str) -> SagaResult:
        """Run the refund, compensating whatever committed if a step fails."""
        return await self.execute(RefundRequest(order=order, amount=amount, reason=reason))

    def idempotency_key(self, request: RefundRequest) -> str:
        """The order plus the amount: a retried refund replays, never pays twice."""
        return f"refund:{request.order.order_id}:{request.amount.cents}"

    def precondition(self, request: RefundRequest) -> str | None:
        """Re-check the authoritative refund policy against fresh repository state."""
        validation = self._refund_policy.validate(
            order_id=request.order.order_id,
            customer_id=request.order.user_id,
            amount=request.amount,
            action="process_refund",
        )
        return None if validation.allowed else validation.reason

    def steps(self, request: RefundRequest) -> list[SagaStep]:
        """The three steps, each with the action that undoes it."""
        order, amount, reason = request.order, request.amount, request.reason
        previous_status = order.refund_status
        refund_id = str(uuid.uuid4())

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
                    f"Your refund of {amount} for order {order.order_id} is on its way.",
                )

        return [
            SagaStep(name="mark_order_refunded", action=mark_refunded, compensation=restore_status),
            SagaStep(name="record_payment", action=record_payment, compensation=reverse_payment),
            # No compensation: a sent notification cannot be unsent, which
            # is exactly why this step runs only after the rest committed.
            SagaStep(name="notify_customer", action=notify, compensation=None),
        ]
