"""LLM-callable tools for the customer support agent.

The refund policy threshold itself is never hardcoded here or in config —
it lives entirely in customer_policy.md's prose, which the LLM reads and
reasons over to decide which tool to call (see build_system_prompt).
"""

import json

from agentic_commons.persistence.database import Database
from agentic_commons.tools.registry import Tool, ToolRegistry
from agentic_commons.tools.types import ToolResult
from customer_support_agent.core.repositories import HitlQueueRepository, OrdersRepository

_ORDER_ID_SCHEMA = {
    "type": "object",
    "properties": {"order_id": {"type": "string", "description": "The order id."}},
    "required": ["order_id"],
}


def build_tool_registry(database: Database) -> ToolRegistry:
    """Build the tool registry backing the customer support agent."""
    orders = OrdersRepository(database)
    hitl_queue = HitlQueueRepository(database)
    registry = ToolRegistry()

    def get_order_status(order_id: str) -> ToolResult:
        order = orders.get_order(order_id)
        if order is None:
            return ToolResult(success=False, output=f"No order found with id {order_id}")
        return ToolResult(
            success=True,
            output=json.dumps(
                {"order_id": order["order_id"], "order_status": order["order_status"]}
            ),
        )

    def get_refund_status(order_id: str) -> ToolResult:
        order = orders.get_order(order_id)
        if order is None:
            return ToolResult(success=False, output=f"No order found with id {order_id}")
        return ToolResult(
            success=True,
            output=json.dumps(
                {
                    "order_id": order["order_id"],
                    "refund_status": order["refund_status"],
                    "amount": order["amount"],
                }
            ),
        )

    def process_refund(order_id: str) -> ToolResult:
        order = orders.get_order(order_id)
        if order is None:
            return ToolResult(success=False, output=f"No order found with id {order_id}")
        if order["refund_status"] == "completed":
            return ToolResult(
                success=False, output=f"Order {order_id} has already been refunded"
            )
        orders.set_refund_status(order_id, "completed")
        return ToolResult(
            success=True,
            output=f"Refund of ${order['amount']:.2f} completed for order {order_id}",
        )

    def queue_for_approval(order_id: str) -> ToolResult:
        order = orders.get_order(order_id)
        if order is None:
            return ToolResult(success=False, output=f"No order found with id {order_id}")
        if order["refund_status"] == "completed":
            return ToolResult(
                success=False, output=f"Order {order_id} has already been refunded"
            )
        orders.set_refund_status(order_id, "pending_review")
        hitl_queue.enqueue(order_id, order["user_id"], order["amount"])
        return ToolResult(
            success=True,
            output=(
                f"Refund of ${order['amount']:.2f} for order {order_id} requires "
                "human approval and has been queued for review"
            ),
        )

    registry.register(
        Tool(
            name="get_order_status",
            description="Get the current shipping/delivery status of an order.",
            parameters=_ORDER_ID_SCHEMA,
            func=get_order_status,
        )
    )
    registry.register(
        Tool(
            name="get_refund_status",
            description="Get the current refund status and amount of an order.",
            parameters=_ORDER_ID_SCHEMA,
            func=get_refund_status,
        )
    )
    registry.register(
        Tool(
            name="process_refund",
            description=(
                "Directly process and complete a refund for an order. "
                "Only for refunds that qualify for automatic approval "
                "per the refund policy."
            ),
            parameters=_ORDER_ID_SCHEMA,
            func=process_refund,
        )
    )
    registry.register(
        Tool(
            name="queue_for_approval",
            description=(
                "Queue a refund for human approval instead of processing it "
                "directly. Use this when the refund policy requires manual review."
            ),
            parameters=_ORDER_ID_SCHEMA,
            func=queue_for_approval,
        )
    )
    return registry
