"""Customer Support's domain tools and execution adapter.

The tool definitions declare the deterministic governance envelope. The
``CustomerSupportToolHandler`` is the agent-owned bridge consumed by the
common Executor; it contains no Gatekeeper logic.
"""

import inspect
from decimal import Decimal
from typing import Annotated, Any

from pydantic import BaseModel, Field, WithJsonSchema

from agentic_commons.executor import ToolExecutionHandler
from agentic_commons.logging import get_logger, instrument
from agentic_commons.tools import ToolCall, ToolDefinition, ToolResult
from customer_support_agent.domain.identity import current_customer_id
from customer_support_agent.domain.models import HitlRequest, Order
from customer_support_agent.domain.refund_policy import RefundPolicy
from customer_support_agent.persistence.repositories import (
    HitlQueueRepository,
    OrdersRepository,
    RefundLedgerRepository,
)
from customer_support_agent.resilience.refund_saga import RefundSaga

_NOT_FOUND = "No order found with id {order_id}"
NAMESPACE = "support"

# An amount as the model sends it: dollars, parsed into an exact Decimal
# (never a float) and advertised as a plain number, which is what models
# fill in reliably. `Money.parse` turns it into cents; a bad precision is
# refused there with a reason, not dropped here as an invalid call.
DollarAmount = Annotated[Decimal, WithJsonSchema({"type": "number"})]


class OrderLookupArgs(BaseModel):
    """Arguments for the two read-only lookups."""

    order_id: int = Field(description="The order identifier, e.g. 10042")


class RefundArgs(BaseModel):
    """Arguments for issuing a refund."""

    order_id: int = Field(description="The order to refund")
    amount: DollarAmount = Field(
        description="Refund amount in dollars (e.g. 137.32): the full order total, confirmed first"
    )
    reason: str = Field(default="", description="Why the refund is being issued")


class ApprovalArgs(BaseModel):
    """Arguments for queuing a refund for human approval."""

    order_id: int = Field(description="The order whose refund needs review")
    amount: DollarAmount = Field(
        description="Refund amount in dollars (e.g. 137.32): the full order total"
    )


@instrument("customer_support_agent")
class SupportTools:
    """The domain toolset, built over injected repositories."""

    def __init__(
        self,
        orders: OrdersRepository,
        hitl: HitlQueueRepository,
        refunds: RefundLedgerRepository,
        saga: RefundSaga,
        refund_policy: RefundPolicy | None = None,
    ) -> None:
        self._orders = orders
        self._refund_policy = refund_policy or RefundPolicy(orders)
        self._hitl = hitl
        self._refunds = refunds
        self._saga = saga
        self._logger = get_logger("customer_support_agent")

    def definitions(self) -> list[ToolDefinition]:
        """Every tool plus the metadata consumed by generic governance."""
        return [
            ToolDefinition(
                name="get_order_status",
                namespace=NAMESPACE,
                description=(
                    "Look up an order: what was bought, its delivery status, "
                    "where the parcel is, its tracking number and estimated "
                    "delivery date, and the total. Use this to answer any "
                    "question about where an order is, what it was, or what it "
                    "cost. Also call it before any refund action, to confirm "
                    "the amount."
                ),
                args_schema=OrderLookupArgs,
                handler=self.get_order_status,
                required_permission="read_orders",
                surface="editable",
                reversible=True,
                side_effecting=False,
            ),
            ToolDefinition(
                name="get_refund_status",
                namespace=NAMESPACE,
                description=(
                    "Check an order's refund state. Only use this when the "
                    "customer has asked about a refund."
                ),
                args_schema=OrderLookupArgs,
                handler=self.get_refund_status,
                required_permission="read_orders",
                surface="editable",
                reversible=True,
                side_effecting=False,
            ),
            ToolDefinition(
                name="process_refund",
                namespace=NAMESPACE,
                description=(
                    "Issue a full refund for an order, within the agent's direct "
                    "refund limit. Partial refunds are not supported."
                ),
                args_schema=RefundArgs,
                handler=self.process_refund,
                required_permission="process_refund",
                surface="editable",
                reversible=False,
                side_effecting=True,
                requires_authority_rank=5,
            ),
            ToolDefinition(
                name="queue_for_approval",
                namespace=NAMESPACE,
                description=(
                    "Place a full refund in the human review queue. Only for an "
                    "order total above the agent's direct refund limit."
                ),
                args_schema=ApprovalArgs,
                handler=self.queue_for_approval,
                required_permission="queue_for_approval",
                surface="append_only",
                reversible=True,
                side_effecting=True,
            ),
        ]

    def _owned_order(self, order_id: int) -> Order | None:
        """The order, if the authenticated customer owns it."""
        return self._orders.get_for_customer(order_id, current_customer_id())

    def get_order_status(self, order_id: int) -> ToolResult:
        """Read one order belonging to the current customer.

        The output is the sentence the model is grounded against, so it
        carries what a customer actually asks about: what the order was,
        where it has got to, and what it cost. A tool that reported only
        a status word left the delivery date as something the model had
        to invent.
        """
        order = self._owned_order(order_id)
        if order is None:
            return ToolResult(success=False, error=_NOT_FOUND.format(order_id=order_id))
        parcel = f", {order.whereabouts}" if order.is_trackable or order.estimated_delivery_date else ""
        tracking = (
            f", tracking {order.carrier} {order.tracking_number}"
            if order.tracking_number and order.carrier
            else ""
        )
        return ToolResult(
            success=True,
            output=(
                f"Order {order.order_id}: {order.description}, {order.order_status}"
                f"{parcel}{tracking}, total {order.amount}"
            ),
            data={**order.model_dump(exclude={"amount", "unit_price"}),
                  "amount": str(order.amount), "unit_price": str(order.unit_price)},
        )

    def get_refund_status(self, order_id: int) -> ToolResult:
        """Read one order's refund state."""
        order = self._owned_order(order_id)
        if order is None:
            return ToolResult(success=False, error=_NOT_FOUND.format(order_id=order_id))
        pending = self._hitl.pending_for_order(order_id)
        detail = " (a review request is already pending)" if pending else ""
        return ToolResult(
            success=True,
            output=f"Order {order.order_id} refund status: {order.refund_status}{detail}",
            data={"refund_status": order.refund_status, "pending_review": pending is not None},
        )

    async def process_refund(self, order_id: int, amount: Any, reason: str = "") -> ToolResult:
        """Issue a refund through the saga with domain-level defense in depth."""
        customer_id = current_customer_id()
        validation = self._refund_policy.validate(
            order_id,
            customer_id,
            amount,
            action="process_refund",
        )
        if not validation.allowed or validation.order is None or validation.amount is None:
            return ToolResult(success=False, error=validation.reason)
        refund = validation.amount
        result = await self._saga.issue(validation.order, refund, reason)
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
            output=f"Refunded {refund} for order {order_id}",
            data={"order_id": order_id, "amount": str(refund), "amount_cents": refund.cents,
                  "steps": result.completed},
        )

    def queue_for_approval(self, order_id: int, amount: Any) -> ToolResult:
        """Enqueue a refund for a human to decide."""
        customer_id = current_customer_id()
        validation = self._refund_policy.validate(
            order_id,
            customer_id,
            amount,
            action="queue_for_approval",
        )
        if not validation.allowed or validation.order is None or validation.amount is None:
            return ToolResult(success=False, error=validation.reason)
        order = validation.order
        requested = validation.amount
        existing = self._hitl.pending_for_order(order_id)
        if existing is not None:
            return ToolResult(
                success=True,
                output=f"Order {order_id} is already awaiting human review",
                data={"already_queued": True},
            )
        self._hitl.save(
            HitlRequest(order_id=order_id, user_id=order.user_id, requested_amount=requested)
        )
        self._orders.set_refund_status(order_id, "pending_review")
        return ToolResult(
            success=True,
            output=(
                f"Refund of {requested} for order {order_id} has been queued for "
                "human review"
            ),
            data={
                "queued": True,
                "request_id": self._hitl.last_id(),
                "order_id": order_id,
                "amount": str(requested),
                "amount_cents": requested.cents,
                "status": "pending_review",
                "action": "queue_for_approval",
            },
        )


class CustomerSupportToolHandler(ToolExecutionHandler):
    """Agent-owned execution adapter for the common Executor."""

    def __init__(self, tools: SupportTools) -> None:
        self._tools = tools
        self._handlers = {
            definition.qualified_name: definition.handler
            for definition in tools.definitions()
        }

    async def execute(self, call: ToolCall) -> ToolResult:
        """Dispatch only to a tool explicitly registered by this agent."""
        handler = self._handlers.get(call.action)
        if handler is None:
            return ToolResult(success=False, error=f"No handler is registered for '{call.action}'")
        result = handler(**call.params)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, ToolResult):
            return result
        return ToolResult(success=True, output=str(result))
