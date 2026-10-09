"""Waiting on a human, inside the run rather than outside it.

A refund or cancellation above the agent's limit is queued for review. Before durable execution
the turn simply ended there, and the reviewer's decision happened in a
different code path with no connection back to the conversation that
asked. Now the run *pauses*: the customer still gets the "under review"
reply this turn (finalize has run), and the thread interrupts after it,
checkpointed, waiting on `hitl:<request id>`. The reviewer's decision
resumes that same thread, which tells the customer what was decided —
one conversation, one audit trail, one run from request to outcome.

The pause is its own node, after `finalize`, for a reason: LangGraph
re-runs the whole node that called `interrupt()` when it resumes. Put in
`execute`, resuming would re-execute every tool call in the turn.
"""

from typing import Any

from langgraph.types import interrupt

from agent_platform.durability import SUMMARY, WAITING_ON
from customer_support_agent.analytics import record_review
from customer_support_agent.graph.nodes._support import NodeSupport
from customer_support_agent.graph.state import AgentState

# Where in state `execute` records a refund it queued for a person.
AWAITING_REVIEW = "awaiting_review"
# The prefix of what an approval wait is waiting on: `hitl:<request id>`.
REVIEW_WAIT_PREFIX = "hitl:"

_APPROVED = (
    "Good news — your refund of {amount} for order {order_id} was approved and is on its way."
)
_DENIED = "Your refund request for order {order_id} was reviewed and not approved: {reason}"
_FAILED = (
    "Your refund for order {order_id} was approved but could not be completed: {reason}. "
    "Nothing was charged or refunded; our team has been notified."
)
_PENDING = "Your refund request for order {order_id} is still under review; we will update you."


def review_wait_key(request_id: object) -> str:
    """What a run waiting on one review request is registered as waiting on."""
    return f"{REVIEW_WAIT_PREFIX}{request_id}"


class ReviewWaitNodes(NodeSupport):
    async def await_approval(self, state: AgentState) -> dict[str, Any]:
        """Interrupt until the reviewer decides; then say what they decided."""
        pending = dict(state.metadata.get(AWAITING_REVIEW) or {})
        request_id = pending.get("request_id")
        decision = interrupt(
            {
                WAITING_ON: review_wait_key(request_id),
                SUMMARY: (
                    f"Human review of a {pending.get('amount')} refund for order "
                    f"{pending.get('order_id')} (request #{request_id})"
                ),
                "order": pending.get("order_id"),
                "amount": pending.get("amount"),
                "request": request_id,
                "resumes_when": f"/queue approve {request_id}  or  /queue deny {request_id}",
            }
        )
        verdict = dict(decision) if isinstance(decision, dict) else {"decision": str(decision)}
        record_review(pending, verdict)
        follow_up = _reply(pending, verdict)
        self._done(
            "review_decision",
            event_kind="escalation",
            summary=f"Reviewer decision: {verdict.get('decision', 'unknown')}",
            fields={**verdict, "order_id": pending.get("order_id"), "reply": follow_up},
        )
        return {
            "metadata": {
                **state.metadata,
                AWAITING_REVIEW: None,
                "review_decision": verdict,
                "follow_up": follow_up,
            }
        }


def _reply(pending: dict[str, Any], verdict: dict[str, Any]) -> str:
    """The customer-facing sentence for a reviewer's decision."""
    values = {
        "order_id": pending.get("order_id"),
        "amount": pending.get("amount"),
        "reason": verdict.get("reason") or "no reason was given",
    }
    outcome = verdict.get("decision")
    if outcome == "approved" and verdict.get("refund_issued"):
        return _APPROVED.format(**values)
    if outcome == "denied":
        return _DENIED.format(**values)
    if outcome in ("approved", "failed"):
        return _FAILED.format(**values)
    return _PENDING.format(**values)
