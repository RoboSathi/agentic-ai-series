"""Customer Support's own console commands (LLD §20, §21).

`/queue` reads this agent's review queue and speaks this
agent's vocabulary, so it belongs to this agent. It used to live in
`agent_platform/commands/`, which meant the platform's console could
not be opened without a refund database behind it.

Scoped like the agent's tools: in a console session, `/queue` shows,
approves and denies only the requests of the customer verified in that
session (`ConsoleScope`), and says "verify first" while nobody is. A
request id that is someone else's gets the same answer as one that does
not exist, so the command cannot be used to learn which requests exist.
Built without a scope (a script, a test), it is the whole queue. The agent
has no route to any of this: commands are dispatched before a message ever
becomes a conversation turn.
"""

from collections.abc import Callable
from typing import Any, get_args

from agent_platform.commands import CommandError, CommandResult
from agent_platform.commands._pool import PoolBackedCommand
from agent_platform.commands.scope import ConsoleScope
from agent_platform.config import get_settings
from agent_platform.durability import DurableAgent
from agent_platform.observability.queue import LittlesLawQueueModel
from agent_platform.persistence import SqliteConnectionPool
from customer_support_agent.config import get_agent_settings
from customer_support_agent.domain.models import OrderStatus
from customer_support_agent.graph.nodes.review_wait import review_wait_key
from customer_support_agent.persistence.repositories import (
    HitlQueueRepository,
)


def _order_statuses() -> frozenset[str]:
    """The statuses an order can hold, from the domain type itself.

    Taken from the `OrderStatus` literal rather than restated here: a
    second list of statuses is a list that goes stale the first time the
    domain gains one.
    """
    return frozenset(get_args(OrderStatus))


_APPROVE = "approve"
_DENY = "deny"


# The decisions that settle a request; anything else decided nothing to report.
_DECIDED = frozenset({"approved", "denied", "failed"})


class QueueCommand(PoolBackedCommand):
    """Show the human review queue and the staffing it implies.

    The queue is this agent's — a refund waiting on a person. The
    staffing arithmetic is not: Little's Law is the platform's, and is
    composed here rather than restated.
    """

    group = "Agent"
    name = "queue"
    summary = "Human review queue: list, approve, deny, staffing maths"
    usage = "/queue [approve <id> | deny <id> [reason] | <arrivals_per_hour> <minutes_each>]"

    def __init__(
        self,
        pool: SqliteConnectionPool | None = None,
        agent: Callable[[], Any] | None = None,
    ) -> None:
        super().__init__(pool)
        self._agent = agent
        self._scope: ConsoleScope | None = None

    def bind_scope(self, scope: ConsoleScope) -> None:
        """Show and decide only the console session's verified customer's requests."""
        self._scope = scope

    async def run(self, args: list[str]) -> CommandResult:
        if self._scope is not None and self._scope.scoped and self._scope.subject() is None:
            return CommandResult(title="Review queue", ok=False).line(
                get_settings().commands.verify_first
            )
        if args and args[0].casefold() in (_APPROVE, _DENY):
            return await self._decide(args)
        return self._overview(args)

    def _customer(self) -> int | None:
        """The verified customer this console is scoped to, or None when unscoped."""
        subject = self._scope.subject() if self._scope is not None else None
        return int(subject) if subject is not None else None

    def _is_theirs(self, request_id: int) -> bool:
        """Whether a request belongs to the scoped customer (always, when unscoped)."""
        customer = self._customer()
        if customer is None:
            return True
        request = HitlQueueRepository(self.pool).get(request_id)
        return request is not None and request.user_id == customer

    async def _decide(self, args: list[str]) -> CommandResult:
        """Apply a reviewer's decision, then resume the turn that was waiting on it.

        The decision itself is the review service's: the same eligibility
        re-check and the same refund saga as ever — one payment path. What
        is new is the second half: the customer's turn paused on this
        request resumes and tells them what was decided.
        """
        from customer_support_agent.graph.builder import build_review_service

        if len(args) < 2 or not args[1].isdigit():
            raise CommandError(f"Usage: {self.usage}")
        request_id = int(args[1])
        if not self._is_theirs(request_id):
            # The answer an unknown id gets: another customer's request is not
            # confirmed to exist.
            return CommandResult(title=f"Review request {request_id} — not_found").line(
                f"Request {request_id} is not pending (already decided, or not found). "
                "/queue lists what is waiting."
            )
        reviewer = get_agent_settings().console_reviewer
        service = build_review_service(self.pool)
        if args[0].casefold() == _APPROVE:
            outcome = await service.approve(request_id, reviewer)
        else:
            outcome = service.deny(request_id, reviewer, " ".join(args[2:]))
        result = CommandResult(title=f"Review request {request_id} — {outcome.decision}")
        if outcome.decision not in _DECIDED:
            return result.line(
                f"Request {request_id} is not pending (already decided, or not found). "
                "/queue lists what is waiting."
            )
        result.line(outcome.reason or f"order {outcome.order_id}, {outcome.amount}")
        return await self._resume_waiting(request_id, outcome.model_dump(mode="json"), result)

    async def _resume_waiting(
        self, request_id: int, decision: dict[str, Any], result: CommandResult
    ) -> CommandResult:
        agent = self._agent() if self._agent is not None else None
        if not isinstance(agent, DurableAgent):
            return result.line("No durable agent is running; no conversation to resume.")
        record = agent.runs.waiting_on(review_wait_key(request_id))
        if record is None:
            return result.line("No paused conversation was waiting on this request.")
        resumed = await agent.resume(record.thread_id, decision)
        result.line(f"Resumed {record.thread_id} → {resumed.status}")
        return result.line(f"ASSISTANT → {resumed.response}")

    def _overview(self, args: list[str]) -> CommandResult:
        customer = self._customer()
        queue = HitlQueueRepository(self.pool)
        pending = queue.pending() if customer is None else queue.pending_for_user(customer)
        result = CommandResult(title=f"Review queue — {len(pending)} pending")
        if pending:
            # Scoped to one customer, a customer column says the same thing every row.
            result.headers = [
                "id",
                "order",
                *(["customer"] if customer is None else []),
                "amount",
                "raised",
            ]
            for request in pending[: get_settings().commands.recent_window]:
                result.row(
                    str(request.id),
                    str(request.order_id),
                    *([str(request.user_id)] if customer is None else []),
                    str(request.requested_amount),
                    request.created_at[:16],
                )
        else:
            result.line("Nothing awaiting review.")

        review = get_settings().observability.review_queue
        arrivals = float(args[0]) if args else max(len(pending), 1) * 1.0
        # The configured review time is the default, not a literal here.
        # A hardcoded 4.0 meant `minutes_per_review` was config nothing read.
        minutes = float(args[1]) if len(args) > 1 else review.minutes_per_review
        capacity = LittlesLawQueueModel().capacity_for(arrivals, minutes)
        result.line("")
        result.line(
            f"Little's Law at {arrivals:g}/hr × {minutes:g} min: "
            f"{capacity.reviewer_hours_per_hour:.2f} reviewer-hours per hour "
            f"({capacity.reviewers_required} people)"
        )
        return result
