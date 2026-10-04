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

from agent_platform.config import get_settings
from agent_platform.guardrails import ResponseGuardrailResult
from agent_platform.logging import get_logger, instrument
from agent_platform.tools import ToolCall, ToolResult
from customer_support_agent.domain.models import REPLY_REVIEW, HitlRequest, ReplyReview
from customer_support_agent.persistence.repositories import HitlQueueRepository
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


@instrument("customer_support_agent")
class ReplyReferral:
    """Queues a reply the hallucination check trimmed or withheld, for a person to judge.

    The same queue as a refund, because it is the same question asked of a
    person: was the agent's harness right to stop here? Approving says the
    check was wrong and sends the customer what it removed; denying says it
    was right. Nothing waits on the answer: the customer already has the
    reply that survived.
    """

    def __init__(self, queue: HitlQueueRepository) -> None:
        self._queue = queue

    def refer(
        self, result: ResponseGuardrailResult, *, customer_id: int | None, session_id: str
    ) -> int | None:
        """Queue the cut reply; None when nothing was cut or no customer is known."""
        if not result.escalated or customer_id is None:
            return None
        reasons = get_settings().guardrails.block_reasons
        self._queue.save(
            HitlRequest(
                user_id=customer_id,
                action=REPLY_REVIEW,
                session_id=session_id,
                detail=ReplyReview(
                    sent=result.response,
                    removed=result.removed_sentences,
                    reason_code=result.reason_code,
                    reason=reasons.get(result.reason_code, result.reason_code),
                    withheld=result.withheld,
                ),
            )
        )
        return self._queue.last_id()
