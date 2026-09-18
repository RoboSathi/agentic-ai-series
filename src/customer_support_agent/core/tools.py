"""The agent's domain tools (LLD §21).

Five tools, each a thin wrapper over a repository or the refund saga,
each declaring its own surface class and blast radius so the Gatekeeper
can reason about it without a separate policy table.

The tools do not authorize themselves. They are registered with
`tools.NamespacedToolRegistry` and executed only through the Gatekeeper —
which is what makes "no execution path bypasses mediation" a property of
the wiring rather than a rule everyone has to remember.
"""

from pydantic import BaseModel, Field

from agentic_commons.logging import get_logger
from agentic_commons.tools import ToolDefinition, ToolResult
from customer_support_agent.core.refund_saga import RefundSaga
from customer_support_agent.core.repositories import (
    HitlQueueRepository,
    OrdersRepository,
    RefundLedgerRepository,
)
from customer_support_agent.core.schema import HitlRequest

NAMESPACE = "support"


class OrderLookupArgs(BaseModel):
    """Arguments for the two read-only lookups."""

    order_id: str = Field(description="The order identifier, e.g. 10042")


class RefundArgs(BaseModel):
    """Arguments for issuing a refund."""

    order_id: str = Field(description="The order to refund")
    amount: float = Field(description="Amount in USD, confirmed from the order first")
    reason: str = Field(default="", description="Why the refund is being issued")


class GoodwillArgs(BaseModel):
    """Arguments for a goodwill credit."""

    user_id: str = Field(description="The customer receiving the credit")
    amount: float = Field(description="Credit amount in USD")
    reason: str = Field(default="", description="Why the credit is being applied")


class ApprovalArgs(BaseModel):
    """Arguments for queuing a refund for human approval."""

    order_id: str = Field(description="The order whose refund needs review")
    amount: float = Field(description="Amount the customer is requesting")


class SupportTools:
    """The domain toolset, built over injected repositories."""

    def __init__(
        self,
        orders: OrdersRepository,
        hitl: HitlQueueRepository,
        refunds: RefundLedgerRepository,
        saga: RefundSaga,
    ) -> None:
        self._orders = orders
        self._hitl = hitl
        self._refunds = refunds
        self._saga = saga
        self._logger = get_logger("customer_support_agent")

    def definitions(self) -> list[ToolDefinition]:
        """Every tool, with the governance properties the Gatekeeper reads."""
        return [
            ToolDefinition(
                name="get_order_status",
                namespace=NAMESPACE,
                description=(
                    "Look up an order's status, refund status and amount. "
                    "Always call this before any refund action."
                ),
                args_schema=OrderLookupArgs,
                handler=self.get_order_status,
                surface="editable",
                blast_radius="single_record",
                reversible=True,
            ),
            ToolDefinition(
                name="get_refund_status",
                namespace=NAMESPACE,
                description="Check whether an order has already been refunded or is under review.",
                args_schema=OrderLookupArgs,
                handler=self.get_refund_status,
                surface="editable",
                blast_radius="single_record",
                reversible=True,
            ),
            ToolDefinition(
                name="process_refund",
                namespace=NAMESPACE,
                description=(
                    "Issue a refund for an order. Only for amounts the policy "
                    "permits to be auto-approved."
                ),
                args_schema=RefundArgs,
                handler=self.process_refund,
                surface="editable",
                blast_radius="account",
                reversible=False,
                requires_authority_rank=5,
            ),
            ToolDefinition(
                name="queue_for_approval",
                namespace=NAMESPACE,
                description=(
                    "Place a refund request in the human review queue. Use this "
                    "for any amount above the auto-approval threshold."
                ),
                args_schema=ApprovalArgs,
                handler=self.queue_for_approval,
                surface="append_only",
                blast_radius="single_record",
                reversible=True,
            ),
            ToolDefinition(
                name="apply_goodwill_credit",
                namespace=NAMESPACE,
                description="Apply a small goodwill credit to a customer's account.",
                args_schema=GoodwillArgs,
                handler=self.apply_goodwill_credit,
                surface="editable",
                blast_radius="account",
                reversible=False,
            ),
        ]

    def get_order_status(self, order_id: str) -> ToolResult:
        """Read one order."""
        order = self._orders.get(order_id)
        if order is None:
            return ToolResult(success=False, error=f"No order found with id {order_id}")
        return ToolResult(
            success=True,
            output=(
                f"Order {order.order_id}: {order.order_status}, "
                f"${order.amount:.2f}, refund status {order.refund_status}"
            ),
            data=order.model_dump(),
        )

    def get_refund_status(self, order_id: str) -> ToolResult:
        """Read one order's refund state."""
        order = self._orders.get(order_id)
        if order is None:
            return ToolResult(success=False, error=f"No order found with id {order_id}")
        pending = self._hitl.pending_for_order(order_id)
        detail = " (a review request is already pending)" if pending else ""
        return ToolResult(
            success=True,
            output=f"Order {order.order_id} refund status: {order.refund_status}{detail}",
            data={"refund_status": order.refund_status, "pending_review": pending is not None},
        )

    async def process_refund(self, order_id: str, amount: float, reason: str = "") -> ToolResult:
        """Issue a refund through the saga.

        The double-refund check lives here rather than in the prompt: a
        policy the model is asked to remember is a policy that fails on
        the turn the model is distracted.
        """
        order = self._orders.get(order_id)
        if order is None:
            return ToolResult(success=False, error=f"No order found with id {order_id}")
        if order.already_refunded:
            return ToolResult(
                success=False,
                error=f"Order {order_id} has already been refunded and cannot be refunded again",
            )
        result = await self._saga.issue(order, amount, reason)
        if not result.succeeded:
            return ToolResult(
                success=False,
                error=(
                    f"Refund failed at '{result.failed_step}' and was rolled back "
                    f"({', '.join(result.compensated) or 'nothing to compensate'})"
                ),
                data=result.model_dump(mode="json"),
            )
        return ToolResult(
            success=True,
            output=f"Refunded ${amount:.2f} for order {order_id}",
            data={"order_id": order_id, "amount": amount, "steps": result.completed},
        )

    def queue_for_approval(self, order_id: str, amount: float) -> ToolResult:
        """Enqueue a refund for a human to decide."""
        order = self._orders.get(order_id)
        if order is None:
            return ToolResult(success=False, error=f"No order found with id {order_id}")
        existing = self._hitl.pending_for_order(order_id)
        if existing is not None:
            return ToolResult(
                success=True,
                output=f"Order {order_id} is already awaiting human review",
                data={"already_queued": True},
            )
        self._hitl.save(
            HitlRequest(order_id=order_id, user_id=order.user_id, requested_amount=amount)
        )
        self._orders.set_refund_status(order_id, "pending_review")
        return ToolResult(
            success=True,
            output=(
                f"Refund of ${amount:.2f} for order {order_id} has been queued for "
                f"human review; the customer will hear back shortly"
            ),
            data={"queued": True, "request_id": self._hitl.last_id()},
        )

    def apply_goodwill_credit(self, user_id: str, amount: float, reason: str = "") -> ToolResult:
        """Apply a goodwill credit, recorded in the refund ledger."""
        self._refunds.record(
            f"goodwill:{user_id}:{amount:.2f}", "", user_id, amount, reason or "goodwill"
        )
        return ToolResult(
            success=True,
            output=f"Applied a ${amount:.2f} goodwill credit to {user_id}",
            data={"user_id": user_id, "amount": amount},
        )
