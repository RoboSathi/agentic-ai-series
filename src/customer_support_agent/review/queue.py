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

from agent_platform.audit import AuditEntry, AuditLedger, NullAuditLedger
from agent_platform.core.primitives.money import Money
from agent_platform.gatekeeper.types import AuthorityContext, Standing
from agent_platform.hallucination_check import TrustCalibrationTracker, TrustReport
from agent_platform.logging import get_logger, logged
from agent_platform.resilience import SagaResult
from agent_platform.security.pii.active import restore_if_on
from customer_support_agent.config import get_agent_settings
from customer_support_agent.domain.models import HitlRequest, Order, ReviewAction
from customer_support_agent.persistence.repositories import HitlQueueRepository, OrdersRepository
from customer_support_agent.resilience.cancellation_saga import CancellationSaga
from customer_support_agent.resilience.refund_saga import RefundSaga


class ReviewOutcome(BaseModel):
    """What a reviewer decided, and what followed from it."""

    request_id: int
    decision: str
    action: ReviewAction = "refund"
    order_id: int | None = None
    amount: Money | None = None
    refund_issued: bool = False
    reason: str = ""
    # A reply review: the conversation it came from, and what the customer is now sent.
    session_id: str | None = None
    removed: list[str] = Field(default_factory=list)
    follow_up: str = ""


class ReviewQueueService:
    """Lists pending refunds and applies a reviewer's decision."""

    def __init__(
        self,
        queue: HitlQueueRepository,
        orders: OrdersRepository,
        saga: RefundSaga,
        audit: AuditLedger | None = None,
        trust: TrustCalibrationTracker | None = None,
        cancellation: CancellationSaga | None = None,
    ) -> None:
        self._queue = queue
        self._orders = orders
        # Whether reviewers trust the agent about as much as they should
        # (E27). A queue where every escalation is waved through is a
        # rubber stamp, and one where every escalation is overturned is a
        # control nobody believes — both are visible only if the
        # agreement rate is counted.
        self._trust = trust or TrustCalibrationTracker()
        self._saga = saga
        # An approved cancellation cancels and refunds as one decision.
        self._cancellation = cancellation or CancellationSaga(orders, saga)
        self._audit = audit or NullAuditLedger()
        self._logger = get_logger("customer_support_agent")

    def pending(self) -> list[HitlRequest]:
        """Everything waiting on a human decision, oldest first."""
        return self._queue.pending()

    @logged("customer_support_agent")
    async def approve(self, request_id: int, reviewer: str) -> ReviewOutcome:
        """Approve a queued refund and issue it through the refund saga."""
        request = self._find(request_id)
        if request is None:
            return ReviewOutcome(
                request_id=request_id, decision="not_found", reason="no pending request"
            )
        if request.is_reply:
            return self._decide_reply(request, reviewer, approved=True)
        order = None if request.order_id is None else self._orders.get(request.order_id)
        if order is None or request.requested_amount is None:
            return ReviewOutcome(
                request_id=request_id,
                decision="failed",
                action=request.action,
                reason=f"order {request.order_id} no longer exists",
            )
        # The eligibility rule is re-checked at approval, not trusted from
        # when it was queued: an order can change state while it waits,
        # and a refund approved against a stale fact is still wrong.
        ineligible = _ineligibility(request, order)
        if ineligible:
            self._queue.decide(request_id, "denied")
            # The agent was wrong to queue this one, and the human caught
            # it: appropriate reliance, which is what the tracker is
            # measuring against automation bias.
            self._trust.record(agent_correct=False, human_accepted=False)
            return ReviewOutcome(
                request_id=request_id,
                decision="denied",
                action=request.action,
                order_id=order.order_id,
                reason=ineligible,
            )
        result = await self._issue(request, order, f"approved by {reviewer}")
        if not result.succeeded:
            return ReviewOutcome(
                request_id=request_id,
                decision="failed",
                action=request.action,
                order_id=order.order_id,
                amount=request.requested_amount,
                reason=f"refund failed at '{result.failed_step}' and was rolled back",
            )
        self._queue.decide(request_id, "approved")
        # The agent escalated something the reviewer agreed with.
        self._trust.record(agent_correct=True, human_accepted=True)
        # RATIFIED: the agent had no authority for this; the reviewer did, and adopted it.
        self._record(request, reviewer, "allow", "approved by reviewer", standing="RATIFIED")
        return ReviewOutcome(
            request_id=request_id,
            decision="approved",
            action=request.action,
            order_id=order.order_id,
            amount=request.requested_amount,
            refund_issued=True,
        )

    async def _issue(self, request: HitlRequest, order: Order, reason: str) -> SagaResult:
        """Run whatever the reviewer approved, through its own saga."""
        amount = request.requested_amount or order.amount
        if request.action == "cancel_and_refund":
            return await self._cancellation.cancel(order, amount, reason)
        return await self._saga.issue(order, amount, reason)

    @logged("customer_support_agent")
    def deny(self, request_id: int, reviewer: str, reason: str = "") -> ReviewOutcome:
        """Refuse a queued refund. No money moves."""
        request = self._find(request_id)
        if request is None:
            return ReviewOutcome(
                request_id=request_id, decision="not_found", reason="no pending request"
            )
        if request.is_reply:
            return self._decide_reply(request, reviewer, approved=False, reason=reason)
        self._queue.decide(request_id, "denied")
        if request.order_id is not None:
            self._orders.set_refund_status(request.order_id, "denied")
        # The reviewer overturned an eligible escalation. Recorded as the
        # agent being right and the human declining: sustained, that is
        # algorithm aversion, and it is only visible if it is counted.
        self._trust.record(agent_correct=True, human_accepted=False)
        self._record(request, reviewer, "deny", reason or "denied by reviewer")
        return ReviewOutcome(
            request_id=request_id,
            decision="denied",
            action=request.action,
            order_id=request.order_id,
            amount=request.requested_amount,
            reason=reason or "denied by reviewer",
        )

    def calibration(self) -> TrustReport:
        """How reviewers have been treating the agent's escalations.

        The two failure modes are opposite and equally expensive: waving
        everything through makes the queue a rubber stamp, and
        overturning everything makes the agent a formality nobody trusts.
        """
        return self._trust.report()

    def _decide_reply(
        self, request: HitlRequest, reviewer: str, *, approved: bool, reason: str = ""
    ) -> ReviewOutcome:
        """Whether the hallucination check was right to cut a reply.

        Approving says it was not: the customer is sent what it removed.
        Denying says it was, and nothing more is sent. Either way the
        agreement is counted, so a check that is overturned more often
        than not shows up as one nobody should trust.
        """
        decision = "approved" if approved else "denied"
        self._queue.decide(int(request.id or 0), decision)
        # Approving overturns the check (it was wrong); denying upholds it.
        self._trust.record(agent_correct=not approved, human_accepted=not approved)
        rationale = reason or f"reply review {decision} by reviewer"
        self._record(request, reviewer, "allow" if approved else "deny", rationale)
        removed = request.detail.removed if request.detail is not None else []
        template = get_agent_settings().review_output.reply_follow_up
        return ReviewOutcome(
            request_id=int(request.id or 0),
            decision=decision,
            action=request.action,
            order_id=request.order_id,
            reason=request.detail.reason if request.detail is not None else rationale,
            session_id=request.session_id,
            removed=removed,
            # What the customer is sent is real text: a person approved it, so its tokens
            # are put back; everywhere else it stays as it was stored.
            follow_up=(
                restore_if_on(template.format(removed=" ".join(removed)))
                if approved and removed
                else ""
            ),
        )

    def _find(self, request_id: int) -> HitlRequest | None:
        """A pending request by id, or None if it is absent or decided."""
        return next((item for item in self._queue.pending() if item.id == request_id), None)

    def _record(
        self,
        request: HitlRequest,
        reviewer: str,
        decision: str,
        rationale: str,
        standing: Standing | None = None,
    ) -> None:
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
                authority_context={
                    **AuthorityContext(actor=reviewer, granted_by="workspace").model_dump(
                        mode="json"
                    ),
                    **({"standing": standing} if standing is not None else {}),
                },
                decision=decision,  # type: ignore[arg-type]
                rationale=rationale,
            )
        )


def _ineligibility(request: HitlRequest, order: Order) -> str:
    """Why the approved action can no longer be carried out, or "" if it still can."""
    if order.already_refunded:
        return order.refund_ineligibility_reason
    if request.action == "cancel_and_refund":
        return order.cancellation_ineligibility_reason
    return order.refund_ineligibility_reason
