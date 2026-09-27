"""Sending an escalated action to a person — the harness's move, not the model's.

The model used to choose between paying a refund and queuing it, which
meant it had to be told the limit that separates the two: in the prompt,
in the tool descriptions, in the knowledge articles. Every copy could
drift from the database, and a model talked out of its instructions could
simply pick the tool that pays.

Now the model has one way to ask for money to move. The Gatekeeper reads
the limit from the persisted policy and, above it, escalates; this module
turns that escalation into a queued review. Nothing the model sees states
the limit, so nothing it sees can be stale — and with the Gatekeeper
withheld there is no escalation, so the same request simply pays. That
difference is the harness.
"""

from agentic_commons.logging import get_logger, instrument
from agentic_commons.tools import ToolCall, ToolResult
from customer_support_agent.tools.definitions import SupportTools

_REFUND = "process_refund"
# A cancellation above the limit is not all-or-nothing: stopping the order
# moves no money, so it happens now, and only its refund waits for a person.
_CANCEL = "cancel_order"


@instrument("customer_support_agent")
class ReviewReferral:
    """Queues an escalated money-moving call for human review."""

    def __init__(self, tools: SupportTools) -> None:
        self._tools = tools
        self._logger = get_logger("customer_support_agent")

    def refer(self, call: ToolCall) -> ToolResult | None:
        """Queue the call for review; None when it is not an action a person reviews."""
        refer = {
            _REFUND: self._tools.queue_for_approval,
            _CANCEL: self._tools.cancel_and_refer_refund,
        }.get(call.action.rsplit(".", maxsplit=1)[-1])
        if refer is None:
            return None
        return refer(call.params["order_id"], call.params["amount"])
