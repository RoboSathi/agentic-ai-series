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
from agentic_commons.resilience import SagaResult
from agentic_commons.tools import ToolCall, ToolDefinition, ToolResult
from customer_support_agent.domain.cancellation_policy import CancellationPolicy
from customer_support_agent.domain.identity import current_customer_id
from customer_support_agent.domain.models import HitlRequest, Order, OrderStatus, ReviewAction
from customer_support_agent.domain.refund_policy import RefundPolicy, RefundValidationResult
from customer_support_agent.persistence.repositories import (
    HitlQueueRepository,
    OrdersRepository,
    RefundLedgerRepository,
)
from customer_support_agent.presentation.orders import customer_orders, describe_orders
from customer_support_agent.resilience.cancellation_saga import CancellationSaga
from customer_support_agent.resilience.refund_saga import RefundSaga

_NOT_FOUND = "No order found with id {order_id}"
_NOT_AUTHENTICATED = "No authenticated customer; nothing can be read or changed"
# How a tool states an order's refund state. The hallucination guard reads
# this field to decide whether a "your refund is done" reply is true, so a
# write must state the state it left behind, not only what it did.
_REFUND_STATUS = "Order {order_id} refund status: {status}"
# A waiting review, as its own sentence so it is its own premise. The earlier
# "(a review request is already pending)" read to NLI as *contradicting* "a
# review request has been submitted" — pending and submitted are the same
# fact in two words, and a parenthesis gave it nothing to hold on to.
_REVIEW_PENDING = (
    "A refund review request for order {order_id} has been submitted and is awaiting a decision"
)
# Each refund state in words. With REFUND_STATUS answered straight from this
# tool, the sentence is the customer's reply; the status line after it is
# what the guard's deterministic refund-state check reads. No arrival date is
# promised: the record holds none.
_REFUND_STATE_SENTENCES: dict[str, str] = {
    "none": "No refund has been requested for order {order_id}",
    "pending_review": _REVIEW_PENDING,
    "approved": "The refund for order {order_id} has been approved",
    "completed": "The refund of {amount} for order {order_id} has been completed",
    "denied": "The refund request for order {order_id} was declined by a reviewer",
}
NAMESPACE = "support"
_CANCELLED: OrderStatus = "cancelled"
_ORDER_CANCELLED = "Order {order_id} has been cancelled"
_NO_LONGER = (
    "Order {order_id} was no longer '{status}' when the cancellation tried to commit; "
    "it may have shipped meanwhile"
)
# What the customer is told when a person must decide. It states the
# amount, which grounds the reply, and never the limit that sent it there.
_REVIEW_QUEUED: dict[str, str] = {
    "refund": "Refund of {amount} for order {order_id} has been queued for human review",
    "cancel_and_refund": (
        "Cancellation of order {order_id} and its {amount} refund have been queued for human review"
    ),
}

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


class CancelArgs(BaseModel):
    """Arguments for cancelling an unshipped order."""

    order_id: int = Field(description="The order to cancel")
    amount: DollarAmount = Field(
        description="The order total in dollars (e.g. 137.32), confirmed first; it is refunded"
    )
    reason: str = Field(default="", description="Why the customer is cancelling")


class ListOrdersArgs(BaseModel):
    """Arguments for listing the customer's own orders."""

    order_status: OrderStatus | None = Field(
        default=None, description="Only orders in this state; omit for every order"
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
        cancellation: CancellationSaga | None = None,
        list_limit: int | None = None,
    ) -> None:
        self._orders = orders
        self._refund_policy = refund_policy or RefundPolicy(orders)
        self._cancellation_policy = CancellationPolicy(orders)
        self._hitl = hitl
        self._refunds = refunds
        self._saga = saga
        self._cancellation = cancellation or CancellationSaga(orders, saga)
        # The most orders one listing returns. None: no cap.
        self._list_limit = list_limit
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
                name="list_orders",
                namespace=NAMESPACE,
                description=(
                    "List the customer's own orders, newest first: id, date, product, "
                    "status and total. Use it when the customer describes an order "
                    "without giving its number, or asks what they have ordered."
                ),
                args_schema=ListOrdersArgs,
                handler=self.list_orders,
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
                    "Refund an order in full. Partial refunds are not supported. "
                    "The result says whether it was refunded or sent for human review."
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
                name="cancel_order",
                namespace=NAMESPACE,
                description=(
                    "Cancel an order that has not shipped yet and refund its full total. "
                    "The result says whether it was cancelled or sent for human review."
                ),
                args_schema=CancelArgs,
                handler=self.cancel_order,
                required_permission="cancel_order",
                surface="editable",
                reversible=False,
                side_effecting=True,
                requires_authority_rank=5,
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
        parcel = (
            f", {order.whereabouts}" if order.is_trackable or order.estimated_delivery_date else ""
        )
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
            data={
                **order.model_dump(exclude={"amount", "unit_price"}),
                "amount": str(order.amount),
                "unit_price": str(order.unit_price),
            },
        )

    def get_refund_status(self, order_id: int) -> ToolResult:
        """Read one order's refund state."""
        order = self._owned_order(order_id)
        if order is None:
            return ToolResult(success=False, error=_NOT_FOUND.format(order_id=order_id))
        pending = self._hitl.pending_for_order(order_id)
        return ToolResult(
            success=True,
            output=(
                f"{_refund_state_sentence(order)}. "
                + _REFUND_STATUS.format(order_id=order.order_id, status=order.refund_status)
            ),
            data={"refund_status": order.refund_status, "pending_review": pending is not None},
        )

    def list_orders(self, order_status: OrderStatus | None = None) -> ToolResult:
        """The current customer's orders, newest first, optionally one state only.

        Scoped by the authenticated customer, never by an argument: there
        is no parameter through which the model could name somebody else.
        """
        customer_id = current_customer_id()
        if customer_id is None:
            return ToolResult(success=False, error=_NOT_AUTHENTICATED)
        found = customer_orders(self._orders, customer_id)
        if order_status is not None:
            found = [order for order in found if order.order_status == order_status]
        shown = found[: self._list_limit] if self._list_limit is not None else found
        return ToolResult(
            success=True,
            output=describe_orders(shown),
            data={
                "count": len(shown),
                "total_count": len(found),
                "orders": [_order_summary(order) for order in shown],
            },
        )

    async def process_refund(self, order_id: int, amount: Any, reason: str = "") -> ToolResult:
        """Issue a refund through the saga with domain-level defense in depth."""
        customer_id = current_customer_id()
        if customer_id is None:
            return ToolResult(success=False, error=_NOT_AUTHENTICATED)
        validation = self._refund_policy.validate(
            order_id, customer_id, amount, action="process_refund"
        )
        if not validation.allowed or validation.order is None or validation.amount is None:
            return ToolResult(success=False, error=validation.reason)
        result = await self._saga.issue(validation.order, validation.amount, reason)
        return self._settled(order_id, validation, result, verb="Refunded")

    async def cancel_order(self, order_id: int, amount: Any, reason: str = "") -> ToolResult:
        """Cancel an unshipped order and refund it, through the cancellation saga."""
        customer_id = current_customer_id()
        if customer_id is None:
            return ToolResult(success=False, error=_NOT_AUTHENTICATED)
        validation = self._cancellation_policy.validate(order_id, customer_id, amount)
        if not validation.allowed or validation.order is None or validation.amount is None:
            return ToolResult(success=False, error=validation.reason)
        result = await self._cancellation.cancel(validation.order, validation.amount, reason)
        return self._settled(order_id, validation, result, verb="Cancelled and refunded")

    def _settled(
        self, order_id: int, validation: RefundValidationResult, result: SagaResult, verb: str
    ) -> ToolResult:
        """Report a money-moving saga: the rollback on failure, the read-back state on success."""
        if not result.succeeded:
            return ToolResult(
                success=False,
                error=(
                    f"{verb} failed at '{result.failed_step}' and was rolled back "
                    f"({', '.join(result.compensated) or 'nothing to compensate'})"
                ),
                data=result.model_dump(mode="json"),
            )
        refund = validation.amount
        assert refund is not None
        status = self._settled_status(order_id)
        return ToolResult(
            success=True,
            output=(
                f"{verb} {refund} for order {order_id}. "
                + _REFUND_STATUS.format(order_id=order_id, status=status)
            ),
            data={
                "order_id": order_id,
                "amount": str(refund),
                "amount_cents": refund.cents,
                "steps": result.completed,
                "refund_status": status,
            },
        )

    def _settled_status(self, order_id: int) -> str:
        """The refund state the record holds after a write: read back, never assumed.

        Without it the only status in this turn's evidence is the read taken
        before the refund, and the guard blocks a true "your refund is done".
        """
        order = self._orders.get(order_id)
        return order.refund_status if order is not None else ""

    def queue_for_approval(
        self, order_id: int, amount: Any, action: ReviewAction = "refund"
    ) -> ToolResult:
        """Enqueue a money-moving action for a human to decide.

        Not offered to the model. The harness calls it when the Gatekeeper
        escalates a refund or a cancellation, so whether a person must
        decide is never the model's choice.
        """
        customer_id = current_customer_id()
        if customer_id is None:
            return ToolResult(success=False, error=_NOT_AUTHENTICATED)
        validation = self._review_validation(order_id, customer_id, amount, action)
        if not validation.allowed or validation.order is None or validation.amount is None:
            return ToolResult(success=False, error=validation.reason)
        requested = validation.amount
        if self._hitl.pending_for_order(order_id) is not None:
            return ToolResult(
                success=True,
                output=(
                    f"Order {order_id} is already awaiting human review. "
                    + _REFUND_STATUS.format(
                        order_id=order_id, status=self._settled_status(order_id)
                    )
                ),
                data={"already_queued": True},
            )
        self._hitl.save(
            HitlRequest(
                order_id=order_id,
                user_id=validation.order.user_id,
                requested_amount=requested,
                action=action,
            )
        )
        self._orders.set_refund_status(order_id, "pending_review")
        status = self._settled_status(order_id)
        return ToolResult(
            success=True,
            output=(
                _REVIEW_QUEUED[action].format(amount=requested, order_id=order_id)
                + ". "
                + _REFUND_STATUS.format(order_id=order_id, status=status)
            ),
            data={
                "queued": True,
                "request_id": self._hitl.last_id(),
                "order_id": order_id,
                "amount": str(requested),
                "amount_cents": requested.cents,
                "status": status,
                "action": action,
            },
        )

    def cancel_and_refer_refund(self, order_id: int, amount: Any) -> ToolResult:
        """Cancel an unshipped order now and send only its refund to a person.

        Not offered to the model. The harness calls it when the Gatekeeper
        escalates a cancellation whose refund is above the automatic limit:
        stopping an order moves no money, so there is nothing for a person
        to decide about it — only the refund waits. If the review cannot be
        queued, the cancellation is undone, so an order is never cancelled
        with its refund dropped on the floor.
        """
        customer_id = current_customer_id()
        if customer_id is None:
            return ToolResult(success=False, error=_NOT_AUTHENTICATED)
        validation = self._cancellation_policy.validate(order_id, customer_id, amount)
        if not validation.allowed or validation.order is None:
            return ToolResult(success=False, error=validation.reason)
        previous = validation.order.order_status
        if not self._orders.set_order_status(order_id, _CANCELLED, expected=previous):
            return ToolResult(
                success=False, error=_NO_LONGER.format(order_id=order_id, status=previous)
            )
        queued = self.queue_for_approval(order_id, amount, action="refund")
        if not queued.success:
            self._orders.set_order_status(order_id, previous, expected=_CANCELLED)
            return queued
        return ToolResult(
            success=True,
            output=f"{_ORDER_CANCELLED.format(order_id=order_id)}. {queued.output}",
            data={**queued.data, "order_status": _CANCELLED},
        )

    def _review_validation(
        self, order_id: int, customer_id: int, amount: Any, action: ReviewAction
    ) -> RefundValidationResult:
        """The domain rules the queued action must satisfy now, and again at approval."""
        if action == "cancel_and_refund":
            return self._cancellation_policy.validate(order_id, customer_id, amount)
        return self._refund_policy.validate(
            order_id, customer_id, amount, action="queue_for_approval"
        )


def _refund_state_sentence(order: Order) -> str:
    """Where this order's refund stands, as a sentence a customer can read."""
    template = _REFUND_STATE_SENTENCES.get(order.refund_status, _REFUND_STATUS)
    return template.format(order_id=order.order_id, amount=order.amount, status=order.refund_status)


def _order_summary(order: Order) -> dict[str, Any]:
    """One listed order, as data: what it is and where it stands — no customer details."""
    return {
        "order_id": order.order_id,
        "date": order.timestamp[:10],
        "product": order.description,
        "order_status": order.order_status,
        "refund_status": order.refund_status,
        "amount": str(order.amount),
    }


class CustomerSupportToolHandler(ToolExecutionHandler):
    """Agent-owned execution adapter for the common Executor."""

    def __init__(self, tools: SupportTools) -> None:
        self._tools = tools
        self._handlers = {
            definition.qualified_name: definition.handler for definition in tools.definitions()
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
