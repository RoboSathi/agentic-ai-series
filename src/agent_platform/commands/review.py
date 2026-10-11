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
from typing import Literal

from agent_platform.audit import SqliteAuditLedger
from agent_platform.commands._pool import PoolBackedCommand
from agent_platform.commands.scope import ConsoleScope
from agent_platform.commands.types import CommandError, CommandResult, CommandTable
from agent_platform.config import get_settings
from agent_platform.observability.queue import LittlesLawQueueModel
from agent_platform.persistence import SqliteConnectionPool
from agent_platform.review import ReviewItem, ReviewQueue
from agent_platform.review.handlers import MemoryReviewHandler, QuarantineHandler
from agent_platform.review.service import EVERYONE, ReviewError, ReviewHandler, ReviewService

_SHOW = "show"
# Where a queue is read, and so how its hint says to decide.
Surface = Literal["console", "terminal"]
CONSOLE: Surface = "console"
TERMINAL: Surface = "terminal"
_DECISIONS = {"approve": "approve", "deny": "deny"}

HandlerFactory = Callable[[SqliteConnectionPool], Sequence[ReviewHandler]]


class QueueCommand(PoolBackedCommand):
    """List, show, approve and deny review items — with a comment, always."""

    group = "Inspect"
    name = "queue"

    def __init__(
        self,
        pool: SqliteConnectionPool | None = None,
        handlers: HandlerFactory | None = None,
        surface: Surface = CONSOLE,
    ) -> None:
        super().__init__(pool)
        self._handlers = handlers
        # Where the list is read: the hint under it says how to decide *there*. A terminal's
        # hint names the agent as `make` knows it (`csa`), which only the caller can say.
        self.surface: Surface = surface
        self.make_agent = ""
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
        for name in dict.fromkeys(item.kind for item in items):
            result.table(self.kind_table(service, name, [i for i in items if i.kind == name]))
        return result.line(self.hint())

    def kind_table(
        self, service: ReviewService, kind: str, items: list[ReviewItem]
    ) -> CommandTable:
        """One kind's pending items, under that kind's own column headings."""
        return CommandTable(
            title=self._words.kind_title.format(kind=kind, n=len(items)),
            headers=service.columns(kind),
            rows=[service.cells(item) for item in items],
        )

    def hint(self) -> str:
        """How to decide, said for where the list is being read."""
        agent = self.make_agent or get_settings().agent.id
        return self._words.hints[self.surface].format(agent=agent)

    def _show(self, args: list[str]) -> CommandResult:
        service = self.service()
        item = service.item(self._id(_SHOW, args), self.subject())
        result = CommandResult(
            title=self._words.show_title.format(id=item.id, kind=item.kind, status=item.status)
        )
        fields = zip(service.columns(item.kind), service.cells(item), strict=True)
        result.table(
            CommandTable(headers=self._words.show_headers, rows=[[k, v] for k, v in fields])
        )
        for line in service.describe(item)[1:]:
            if line:
                result.line(line)
        return result.line(self.hint())

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
