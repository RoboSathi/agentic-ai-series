"""`/queue` — the one human-review queue, for every agent (ADR 0012).

Lists what is waiting on a person as a tree by kind, shows one item, and
approves or denies it with a comment the service requires and audits.
The platform's own kinds (a quarantine, a fact or skill memory would not
take on its own word) are always here; an agent adds its own by
registering this command with its handlers (Customer Support's refunds
and cut replies).

Scoped like every console command that reads people's records: in a
console where the agent can say who is verified, only that subject's
items are shown and decided — and, before anyone is verified, only the
agent's own (subject-less) items. Another subject's item gets the answer
a missing one gets. An agent that has no subjects sees everything.

With numbers instead, it is the staffing arithmetic the queue implies
(Little's Law): arrivals per hour, and minutes per review (configured
`observability.review_queue.minutes_per_review` when not given).
"""

from collections.abc import Callable, Sequence

from agent_platform.audit import SqliteAuditLedger
from agent_platform.commands._pool import PoolBackedCommand
from agent_platform.commands.scope import ConsoleScope
from agent_platform.commands.types import CommandError, CommandResult
from agent_platform.config import get_settings
from agent_platform.observability.queue import LittlesLawQueueModel
from agent_platform.persistence import SqliteConnectionPool
from agent_platform.presentation.text_tree import TreeNode, tree_lines
from agent_platform.review import ReviewItem, ReviewQueue
from agent_platform.review.handlers import MemoryReviewHandler, QuarantineHandler
from agent_platform.review.service import EVERYONE, ReviewError, ReviewHandler, ReviewService

_SHOW = "show"
_DECISIONS = {"approve": "approve", "deny": "deny"}

HandlerFactory = Callable[[SqliteConnectionPool], Sequence[ReviewHandler]]


class QueueCommand(PoolBackedCommand):
    """List, show, approve and deny review items — with a comment, always."""

    group = "Inspect"
    name = "queue"

    def __init__(
        self, pool: SqliteConnectionPool | None = None, handlers: HandlerFactory | None = None
    ) -> None:
        super().__init__(pool)
        self._handlers = handlers
        self._scope: ConsoleScope | None = None
        self._words = get_settings().review.words
        self.summary = self._words.summary
        self.usage = self._words.usage

    def bind_scope(self, scope: ConsoleScope) -> None:
        """Show and decide only what the console's scope allows."""
        self._scope = scope

    def subject(self) -> str | None:
        """Whose items this console may see: one subject, the agent's own (""), or all."""
        if self._scope is None or not self._scope.scoped:
            return EVERYONE
        return self._scope.subject() or ""

    def service(self) -> ReviewService:
        """The review service over this agent's queue, with every handler it has."""
        from agent_platform.host.memory import governed_memory

        own = list(self._handlers(self.pool)) if self._handlers is not None else []
        pipeline = governed_memory(self.pool, redact=False).pipeline
        handlers: list[ReviewHandler] = [QuarantineHandler(), MemoryReviewHandler(pipeline), *own]
        # The agent's own ledger: written from a console or a terminal alike without
        # contending for the shared one, and read back by `/audit` and `make audit`.
        ledger = SqliteAuditLedger(self.pool, agent_id=get_settings().agent.id)
        return ReviewService(ReviewQueue(self.pool), handlers, ledger)

    async def run(self, args: list[str]) -> CommandResult:
        action = args[0].casefold() if args else ""
        if action in _DECISIONS:
            return await self._decide(_DECISIONS[action], args[1:])
        if action == _SHOW:
            return self._show(args[1:])
        if args and len(args) <= 2 and all(_number(arg) for arg in args):
            default = get_settings().observability.review_queue.minutes_per_review
            return _staffing(float(args[0]), float(args[1]) if len(args) > 1 else default)
        return self._list(args[0] if args else None)

    def _list(self, kind: str | None) -> CommandResult:
        service = self.service()
        if kind is not None and kind not in service.kinds:
            raise CommandError(
                self._words.unknown_kind.format(kind=kind, kinds=", ".join(service.kinds))
            )
        items = service.visible(self.subject(), kind)
        result = CommandResult(title=self._words.title.format(n=len(items)))
        if not items:
            return result.line(self._words.empty)
        result.lines = tree_lines(self._tree(service, items, result.title))[1:]
        return result.line(self._words.hint)

    def _tree(self, service: ReviewService, items: list[ReviewItem], title: str) -> TreeNode:
        root = TreeNode(title)
        for kind in dict.fromkeys(item.kind for item in items):
            mine = [item for item in items if item.kind == kind]
            branch = root.add(self._words.kind_branch.format(kind=kind, n=len(mine)))
            for item in mine:
                branch.add(self.line(service, item))
        return root

    def line(self, service: ReviewService, item: ReviewItem) -> str:
        """One item as a list shows it: its id, what it asks, and whose it is."""
        described = service.describe(item)
        summary = described[0] if described and described[0] else item.summary
        subject = (
            self._words.subject_part.format(subject=item.subject_id) if item.subject_id else ""
        )
        return self._words.item_line.format(id=item.id, summary=summary) + subject

    def _show(self, args: list[str]) -> CommandResult:
        service = self.service()
        item = service.item(self._id(_SHOW, args), self.subject())
        result = CommandResult(
            title=self._words.show_title.format(id=item.id, kind=item.kind, status=item.status)
        )
        for line in service.describe(item):
            if line:
                result.line(line)
        return result

    async def _decide(self, decision: str, args: list[str]) -> CommandResult:
        item_id = self._id(decision, args)
        reviewer = get_settings().review.console_reviewer
        try:
            outcome = await self.service().decide(
                item_id,
                "approve" if decision == "approve" else "deny",
                reviewer,
                " ".join(args[1:]),
                self.subject(),
            )
        except ReviewError as error:
            raise CommandError(str(error)) from error
        title = self._words.decided.format(
            id=item_id, kind=outcome.kind, status=outcome.status, reviewer=reviewer,
            comment=" ".join(args[1:]),
        )  # fmt: skip
        result = CommandResult(title=title, ok=outcome.status != "failed")
        for line in outcome.lines:
            result.line(line)
        return result

    def _id(self, action: str, args: list[str]) -> int:
        if not args or not args[0].isdigit():
            raise CommandError(self._words.needs_id.format(action=action, usage=self.usage))
        return int(args[0])


def _number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def _staffing(arrivals: float, minutes: float) -> CommandResult:
    """Little's Law: how much reviewing the queue's arrival rate implies."""
    capacity = LittlesLawQueueModel().capacity_for(arrivals, minutes)
    return CommandResult(title=f"Little's Law at {arrivals:g}/hr × {minutes:g} min").line(
        f"{capacity.reviewer_hours_per_hour:.2f} reviewer-hours per hour "
        f"({capacity.reviewers_required} people)"
    )
