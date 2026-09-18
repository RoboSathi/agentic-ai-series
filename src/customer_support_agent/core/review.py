"""The human side of the escalation path (LLD §21, E38).

An escalation that nothing can drain is not a control — it is a way of
losing requests politely. The agent could enqueue refunds for human
approval and no component existed to approve them, so every escalated
refund stopped there permanently.

This is deliberately *not* an agent tool. `approve_hitl_request` is a
`human_controlled` surface, which means exactly what it says: the agent
may propose, and only a person disposes. So the entry point lives here,
for an operator console to call, and the agent has no path to it.

Approving is the same refund the agent would have issued, through the
same saga — a second payment path would be a second set of bugs, and
only one of them would be tested.
"""

from pydantic import BaseModel, Field

from agentic_commons.audit import AuditEntry, AuditLedger, NullAuditLedger
from agentic_commons.gatekeeper.types import AuthorityContext
from agentic_commons.logging import get_logger, logged
from agentic_commons.observability.types import utc_now
from customer_support_agent.core.refund_saga import RefundSaga
from customer_support_agent.core.repositories import HitlQueueRepository, OrdersRepository
from customer_support_agent.core.schema import HitlRequest


class ReviewOutcome(BaseModel):
    """What a reviewer decided, and what followed from it."""

    request_id: int
    decision: str
    order_id: str = ""
    amount: float = 0.0
    refund_issued: bool = False
    reason: str = ""


class ReviewQueueService:
    """Lists pending refunds and applies a reviewer's decision."""

    def __init__(
        self,
        queue: HitlQueueRepository,
        orders: OrdersRepository,
        saga: RefundSaga,
        audit: AuditLedger | None = None,
    ) -> None:
        self._queue = queue
        self._orders = orders
        self._saga = saga
        self._audit = audit or NullAuditLedger()
        self._logger = get_logger("customer_support_agent")

    def pending(self) -> list[HitlRequest]:
        """Everything waiting on a human decision, oldest first."""
        return self._queue.pending()

    def pending_count(self) -> int:
        """Queue depth — the input to the Little's Law staffing model."""
        return len(self._queue.pending())

    @logged("customer_support_agent")
    async def approve(self, request_id: int, reviewer: str) -> ReviewOutcome:
        """Approve a queued refund and issue it through the refund saga."""
        request = self._find(request_id)
        if request is None:
            return ReviewOutcome(
                request_id=request_id, decision="not_found", reason="no pending request"
            )
        order = self._orders.get(request.order_id)
        if order is None:
            return ReviewOutcome(
                request_id=request_id,
                decision="failed",
                reason=f"order {request.order_id} no longer exists",
            )
        # The eligibility rule is re-checked at approval, not trusted from
        # when it was queued: an order can change state while it waits,
        # and a refund approved against a stale fact is still wrong.
        if not order.is_refund_eligible or order.already_refunded:
            self._queue.decide(request_id, "denied")
            return ReviewOutcome(
                request_id=request_id,
                decision="denied",
                order_id=order.order_id,
                reason=order.refund_ineligibility_reason,
            )
        result = await self._saga.issue(order, request.requested_amount, f"approved by {reviewer}")
        if not result.succeeded:
            return ReviewOutcome(
                request_id=request_id,
                decision="failed",
                order_id=order.order_id,
                amount=request.requested_amount,
                reason=f"refund failed at '{result.failed_step}' and was rolled back",
            )
        self._queue.decide(request_id, "approved")
        self._record(request, reviewer, "allow", "approved by reviewer")
        return ReviewOutcome(
            request_id=request_id,
            decision="approved",
            order_id=order.order_id,
            amount=request.requested_amount,
            refund_issued=True,
        )

    @logged("customer_support_agent")
    def deny(self, request_id: int, reviewer: str, reason: str = "") -> ReviewOutcome:
        """Refuse a queued refund. No money moves."""
        request = self._find(request_id)
        if request is None:
            return ReviewOutcome(
                request_id=request_id, decision="not_found", reason="no pending request"
            )
        self._queue.decide(request_id, "denied")
        self._orders.set_refund_status(request.order_id, "denied")
        self._record(request, reviewer, "deny", reason or "denied by reviewer")
        return ReviewOutcome(
            request_id=request_id,
            decision="denied",
            order_id=request.order_id,
            amount=request.requested_amount,
            reason=reason or "denied by reviewer",
        )

    def _find(self, request_id: int) -> HitlRequest | None:
        """A pending request by id, or None if it is absent or decided."""
        return next((item for item in self._queue.pending() if item.id == request_id), None)

    def _record(self, request: HitlRequest, reviewer: str, decision: str, rationale: str) -> None:
        """Audit the human decision under the reviewer's own authority.

        The reviewer is the actor, not the agent: the whole point of the
        escalation is that a person took responsibility, and an audit
        trail that credits the agent loses exactly that.
        """
        self._audit.record(
            AuditEntry(
                trace_id=f"review-{request.id}",
                actor=reviewer,
                action="approve_hitl_request",
                authority_context=AuthorityContext(
                    actor=reviewer, granted_by="workspace", roles=["support_supervisor"]
                ).model_dump(mode="json"),
                decision=decision,  # type: ignore[arg-type]
                rationale=rationale,
            )
        )


class ReviewQueueMetrics(BaseModel):
    """Queue health, for the SLO and staffing views (E38)."""

    pending: int = 0
    oldest_wait_hours: float = 0.0
    measured_at: str = Field(default_factory=lambda: utc_now().isoformat())
