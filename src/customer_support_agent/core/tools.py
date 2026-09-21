"""Customer Support's domain tools and execution adapter.

The tool definitions declare the deterministic governance envelope. The
``CustomerSupportToolHandler`` is the agent-owned bridge consumed by the
common Executor; it contains no Gatekeeper logic.
"""

import inspect

from pydantic import BaseModel, Field

from agentic_commons.executor import ToolExecutionHandler
from agentic_commons.logging import get_logger, instrument
from agentic_commons.tools import PathCardinality, ToolCall, ToolDefinition, ToolResult
from customer_support_agent.core.identity import current_customer_id
from customer_support_agent.core.refund_saga import RefundSaga
from customer_support_agent.core.repositories import (
    HitlQueueRepository,
    OrdersRepository,
    RefundLedgerRepository,
)
from customer_support_agent.core.schema import HitlRequest, Order

_NOT_FOUND = "No order found with id {order_id}"
NAMESPACE = "support"


class OrderLookupArgs(BaseModel):
    """Arguments for the two read-only lookups."""

    order_id: int = Field(description="The order identifier, e.g. 10042")


class RefundArgs(BaseModel):
    """Arguments for issuing a refund."""

    order_id: int = Field(description="The order to refund")
    amount: float = Field(description="Amount in USD, confirmed from the order first")
    reason: str = Field(default="", description="Why the refund is being issued")


class ApprovalArgs(BaseModel):
    """Arguments for queuing a refund for human approval."""

    order_id: int = Field(description="The order whose refund needs review")
    amount: float = Field(description="Amount the customer is requesting")


@instrument("customer_support_agent")
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
        """Every tool plus the metadata consumed by generic governance."""
        return [
            ToolDefinition(
                name="get_order_status",
                namespace=NAMESPACE,
                description=(
                    "Look up an order's delivery status and total. Use this to "
                    "answer any question about where an order is or what it cost. "
                    "Also call it before any refund action, to confirm the amount."
                ),
                args_schema=OrderLookupArgs,
                handler=self.get_order_status,
                required_permission="read_orders",
                surface="editable",
                blast_radius="single_record",
                path_cardinality="single",
                certifiable=True,
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
                blast_radius="single_record",
                path_cardinality="single",
                certifiable=True,
                reversible=True,
                side_effecting=False,
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
                required_permission="process_refund",
                surface="editable",
                blast_radius="account",
                path_cardinality="single",
                certifiable=True,
                reversible=False,
                side_effecting=True,
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
                required_permission="queue_for_approval",
                surface="append_only",
                blast_radius="single_record",
                path_cardinality="single",
                certifiable=True,
                reversible=True,
                side_effecting=True,
            ),
        ]

    def _owned_order(self, order_id: int) -> Order | None:
        """The order, if the authenticated customer owns it."""
        return self._orders.get_for_customer(order_id, current_customer_id())

    def get_order_status(self, order_id: int) -> ToolResult:
        """Read one order belonging to the current customer."""
        order = self._owned_order(order_id)
        if order is None:
            return ToolResult(success=False, error=_NOT_FOUND.format(order_id=order_id))
        return ToolResult(
            success=True,
            output=(f"Order {order.order_id}: {order.order_status}, total ${order.amount:.2f}"),
            data=order.model_dump(),
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

    async def process_refund(self, order_id: int, amount: float, reason: str = "") -> ToolResult:
        """Issue a refund through the saga with domain-level defense in depth."""
        order = self._owned_order(order_id)
        if order is None:
            return ToolResult(success=False, error=_NOT_FOUND.format(order_id=order_id))
        if not order.is_refund_eligible or order.already_refunded:
            return ToolResult(success=False, error=order.refund_ineligibility_reason)
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

    def queue_for_approval(self, order_id: int, amount: float) -> ToolResult:
        """Enqueue a refund for a human to decide."""
        order = self._owned_order(order_id)
        if order is None:
            return ToolResult(success=False, error=_NOT_FOUND.format(order_id=order_id))
        if not order.is_refund_eligible or order.already_refunded:
            return ToolResult(success=False, error=order.refund_ineligibility_reason)
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
                "human review; the customer will hear back shortly"
            ),
            data={"queued": True, "request_id": self._hitl.last_id(), "order_id": order_id, "amount": amount},
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
