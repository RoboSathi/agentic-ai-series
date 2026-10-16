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
from agent_platform.tools import ToolCall
from customer_support_agent.config import get_agent_settings
from customer_support_agent.domain.models import REPLY_REVIEW, HitlRequest, ReplyReview
from customer_support_agent.persistence.repositories import HitlQueueRepository
from customer_support_agent.tools.definitions import (
    CANCEL_AND_REFER_REFUND,
    NAMESPACE,
    QUEUE_FOR_APPROVAL,
)

_REFUND = "process_refund"
# A cancellation above the limit is not all-or-nothing: stopping the order
# moves no money, so it happens now, and only its refund waits for a person.
_CANCEL = "cancel_order"
# Each escalated action and the harness write that sends it to a person.
_REFERRALS = {_REFUND: QUEUE_FOR_APPROVAL, _CANCEL: CANCEL_AND_REFER_REFUND}


@instrument("customer_support_agent")
class ReviewReferral:
    """Turns an escalated money-moving call into the harness's referral call.

    It decides *what* a person is asked to review; it does not write the
    request. The referral call goes through the Executor like every other
    side effect, so the queue write has the same idempotency, deadline and
    trace (EXECUTOR, TOOL RESULT) as the calls around it.
    """

    def __init__(self) -> None:
        self._logger = get_logger("customer_support_agent")

    def referral_call(self, call: ToolCall) -> ToolCall | None:
        """The queueing write for this escalated call; None when no person reviews it."""
        action = _REFERRALS.get(call.action.rsplit(".", maxsplit=1)[-1])
        if action is None:
            return None
        return ToolCall(
            action=f"{NAMESPACE}.{action}",
            params={"order_id": call.params["order_id"], "amount": call.params["amount"]},
            surface=call.surface,
            reversible=False,
            side_effecting=True,
            requires_authority_rank=call.requires_authority_rank,
            required_permission=call.required_permission,
            call_id=call.call_id,
        )


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
        self._app_config = get_settings()
        self._agent_config = get_agent_settings()
        self._queue = queue

    def refer(
        self, result: ResponseGuardrailResult, *, customer_id: int | None, session_id: str
    ) -> int | None:
        """Queue the cut reply; None when nothing was cut or no customer is known."""
        if not result.escalated or customer_id is None:
            return None
        reasons = self._app_config.guardrails.block_reasons
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
                reason=self._agent_config.review_reasons.for_finding(result.reason_code),
            )
        )
        return self._queue.last_id()
