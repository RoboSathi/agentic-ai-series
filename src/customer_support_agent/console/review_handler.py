"""The support agent's review kinds, decided through the platform's one queue (ADR 0012).

A refund, a cancellation's refund and a reply the hallucination check cut
are this agent's own kinds of review item. What deciding one *does* is
unchanged — the review service re-checks eligibility and pays through the
same saga, or sends the customer the sentences the check removed — and
the turn that was waiting on the decision resumes and says what was
decided. What is new is the way in: `/review` and the Review tab, with a
comment the platform requires and audits.
"""

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from agent_platform.config import get_settings
from agent_platform.core.persistence import SqliteConnectionPool
from agent_platform.durability import DurableAgent
from agent_platform.interface.commands.review import QueueCommand
from agent_platform.interface.presentation.text_tree import value_lines
from agent_platform.logging import get_logger
from agent_platform.review import ReviewItem, ReviewOutcome
from agent_platform.review.handlers import none, raised
from customer_support_agent.config import get_agent_settings
from customer_support_agent.domain.models import REPLY_REVIEW, HitlRequest, ReviewAction
from customer_support_agent.graph.nodes.review_wait import review_wait_key
from customer_support_agent.persistence.repositories import HitlQueueRepository, UsersRepository
from customer_support_agent.review.queue import ReviewOutcome as SupportOutcome
from customer_support_agent.review.queue import ReviewQueueService

# The decisions that settle a request, and the review status each one is.
_STATUS = {"approved": "approved", "denied": "denied", "failed": "failed"}


@runtime_checkable
class FollowUpRecipient(Protocol):
    """An agent a reviewer's decision can be reported back through."""

    def deliver_follow_up(self, session_id: str, reply: str) -> None: ...


class SupportReviewHandler:
    """Approves and denies this agent's escalations, then tells the customer."""

    def __init__(
        self,
        pool: SqliteConnectionPool,
        service: Callable[[SqliteConnectionPool], ReviewQueueService],
        agent: Callable[[], Any] | None = None,
    ) -> None:
        self._app_config = get_settings()
        self._agent_config = get_agent_settings()
        self._pool = pool
        self._service = service
        self._agent = agent
        self._logger = get_logger("customer_support_agent")

    @property
    def kinds(self) -> frozenset[str]:
        return frozenset(ReviewAction.__args__)  # type: ignore[attr-defined]

    def describe(self, item: ReviewItem) -> list[str]:
        request = HitlQueueRepository(self._pool).get(int(item.id or 0))
        if request is None:
            return [item.summary]
        sent = request.detail.sent if request.detail is not None else ""
        return [request_line(request), request.reason, *([sent] if sent else [])]

    def columns(self, kind: str) -> list[str]:
        return list(self._app_config.review.words.columns[kind])

    def cells(self, item: ReviewItem) -> list[str]:
        """A refund as its order, amount and customer; a cut reply as what was removed."""
        request = HitlQueueRepository(self._pool).get(int(item.id or 0))
        if request is None:
            return [str(item.id), *[none()] * (len(self.columns(item.kind)) - 1)]
        customer = self._customer(request.user_id)
        if request.is_reply:
            removed = " ".join(request.detail.removed) if request.detail is not None else ""
            why = request.detail.reason if request.detail is not None else request.reason
            return [str(item.id), customer, request.session_id or none(), removed or none(),
                    why or none(), raised(item)]  # fmt: skip
        return [
            str(item.id),
            str(request.order_id) if request.order_id is not None else none(),
            str(request.requested_amount) if request.requested_amount is not None else none(),
            customer,
            request.reason or none(),
            raised(item),
        ]

    def _customer(self, user_id: int) -> str:
        """`Name (#id)` when the customer is on record; the id alone otherwise."""
        user = UsersRepository(self._pool).get(user_id)
        words = self._agent_config.review_output
        return words.customer_cell.format(name=user.name, id=user_id) if user else str(user_id)

    async def approve(self, item: ReviewItem, reviewer: str, comment: str) -> ReviewOutcome:
        outcome = await self._service(self._pool).approve(int(item.id or 0), reviewer)
        return await self._reported(item, outcome, reviewer)

    async def deny(self, item: ReviewItem, reviewer: str, comment: str) -> ReviewOutcome:
        outcome = self._service(self._pool).deny(int(item.id or 0), reviewer, comment)
        return await self._reported(item, outcome, reviewer)

    async def _reported(
        self, item: ReviewItem, outcome: SupportOutcome, reviewer: str
    ) -> ReviewOutcome:
        if outcome.decision not in _STATUS:
            # Decided nothing (no longer pending): nobody is told, and the reviewer is told why.
            not_pending = self._agent_config.review_output.not_pending.format(id=item.id)
            return ReviewOutcome.model_validate(
                {"item_id": item.id, "kind": item.kind, "status": "failed", "lines": [not_pending]}
            )
        told = await self._tell_customer(outcome)
        return ReviewOutcome.model_validate(
            {
                "item_id": item.id,
                "kind": item.kind,
                "status": _STATUS.get(outcome.decision, "failed"),
                "lines": value_lines(decision_tree(outcome, reviewer, told)),
                "effects": {"customer_told": told} if told else {},
            }
        )

    async def _tell_customer(self, outcome: SupportOutcome) -> str:
        """What the customer was sent about this decision, or "" when nobody could be told.

        A refund's paused run resumes and says what was decided; an approved
        reply review is added to its conversation. The run's thread id is in
        the log, not here: the operator decided a request, not a thread.
        """
        agent = self._agent() if self._agent is not None else None
        if outcome.action == REPLY_REVIEW:
            if not outcome.follow_up or not isinstance(agent, FollowUpRecipient):
                return ""
            agent.deliver_follow_up(str(outcome.session_id or ""), outcome.follow_up)
            return outcome.follow_up
        if not isinstance(agent, DurableAgent):
            return ""
        record = agent.runs.waiting_on(review_wait_key(outcome.request_id))
        if record is None:
            return ""
        resumed = await agent.resume(record.thread_id, outcome.model_dump(mode="json"))
        self._logger.info(
            "Review decision resumed a paused run",
            request_id=outcome.request_id,
            thread_id=record.thread_id,
            status=resumed.status,
        )
        return resumed.response


def decision_tree(outcome: SupportOutcome, reviewer: str, told: str) -> dict[str, str]:
    """`request`, `decided`, `effect`, `customer`: one decision as the console reads it."""
    words = get_agent_settings().review_output
    labels = words.labels
    effect_key = (
        "failed" if outcome.decision == "failed" else f"{outcome.action}_{outcome.decision}"
    )
    nothing_to_tell = outcome.action == REPLY_REVIEW and outcome.decision != "approved"
    return {
        labels["request"]: words.request[outcome.action].format(
            order_id=outcome.order_id,
            amount=outcome.amount,
            reason=outcome.reason,
            removed=" ".join(outcome.removed),
        ),
        labels["decided"]: words.decided.format(decision=outcome.decision, reviewer=reviewer),
        labels["effect"]: words.effects.get(effect_key, words.effects["failed"]).format(
            reason=outcome.reason
        ),
        **(
            {}
            if nothing_to_tell
            else {
                labels["customer"]: words.customer_told.format(reply=told)
                if told
                else words.customer_not_told
            }
        ),
    }


def request_line(request: HitlRequest) -> str:
    """What one queued request asks a person to decide, in this agent's words."""
    detail = request.detail
    return (
        get_agent_settings()
        .review_output.request[request.action]
        .format(
            order_id=request.order_id,
            amount=request.requested_amount,
            reason=detail.reason if detail is not None else "",
            removed=" ".join(detail.removed) if detail is not None else "",
        )
    )


def support_queue_command(
    pool: SqliteConnectionPool | None = None, agent: Callable[[], Any] | None = None
) -> QueueCommand:
    """The platform's `/review`, with this agent's refunds and cut replies among its kinds."""

    def handlers(connection: SqliteConnectionPool) -> list[SupportReviewHandler]:
        return [SupportReviewHandler(connection, _review_service, agent)]

    return QueueCommand(pool, handlers=handlers)


def _review_service(pool: SqliteConnectionPool) -> ReviewQueueService:
    """The refund and reply review service, looked up when a decision is made."""
    from customer_support_agent.graph import builder

    return builder.build_review_service(pool)
