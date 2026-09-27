"""Commands that read what the platform itself stores (LLD §20).

Memory is a platform subsystem — every agent built on this harness has
the same four CoALA stores — so `/memory` belongs here.

Commands that read a *product's* tables do not. `/queue` used to live
in this module, and required one particular product's database to exist
before the console could open at all. It now belongs to the agent that
has a review queue, and reaches the console through its `AgentHost`.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

from agentic_commons.commands._pool import PoolBackedCommand
from agentic_commons.commands.types import CommandResult
from agentic_commons.config import get_settings
from agentic_commons.persistence import SqliteConnectionPool
from agentic_commons.text_tree import tree_lines

if TYPE_CHECKING:
    from agentic_commons.memory import WorkingMemoryStore
    from agentic_commons.memory.consolidation import ConsolidationService


class MemoryCommand(PoolBackedCommand):
    """Show what the CoALA stores hold — long-term, or this conversation's.

    Working memory lives in the running process, not the database, so
    the console has to be handed the live facade to show it. Without
    one, `/memory working` says so rather than reporting an empty window
    that would read as "nothing was remembered".
    """

    group = "Inspect"
    name = "memory"
    summary = "What memory holds, as a tree (add 'working' for this conversation)"
    usage = "/memory [working|customer <id>|<search term>]"

    _WORKING = "working"
    _CUSTOMER = "customer"

    def __init__(
        self,
        pool: SqliteConnectionPool | None = None,
        working_memory: "WorkingMemoryStore | None" = None,
        session_id: str = "",
    ) -> None:
        super().__init__(pool)
        self._working_memory = working_memory
        self._session_id = session_id

    async def run(self, args: list[str]) -> CommandResult:
        """The same tree `make show-memory` prints, narrowed by customer or search term."""
        from agentic_commons.memory.inspect import snapshot_tree, take_snapshot
        from agentic_commons.persistence import as_platform_store

        settings = get_settings()
        if args and args[0].casefold() == self._WORKING:
            return self._working_window(settings.commands.value_preview_chars)
        subject, term = self._narrowing(args)
        snapshot = take_snapshot(
            as_platform_store(self.pool), settings.memory.episodic.resolved_path, subject, term
        )
        tree = snapshot_tree(
            snapshot,
            settings.commands.memory_tree_events_per_session,
            settings.commands.tree_preview_chars,
        )
        result = CommandResult(title=f"Memory — {snapshot.headline}")
        result.lines = tree_lines(tree)
        result.line("Use /memory working to see what this conversation is holding.")
        return result

    def _narrowing(self, args: list[str]) -> tuple[str | None, str | None]:
        """`customer <id>` narrows to one customer; anything else is a search term."""
        if len(args) >= 2 and args[0].casefold() == self._CUSTOMER:
            return args[1], " ".join(args[2:]) or None
        return None, " ".join(args) or None

    def _working_window(self, preview: int) -> CommandResult:
        """This session's active window — what the next turn will see."""
        if self._working_memory is None:
            return CommandResult(title="Working memory", ok=False).line(
                "No live agent is attached to this console."
            )
        memory = self._working_memory.get(self._session_id)
        result = CommandResult(
            title=f"Working memory — {len(memory.state)} state item(s), {len(memory.history) // 2} conversation turn(s)"
        )
        if not memory.state and not memory.history:
            return result.line("Nothing established yet. Verify, and it fills as you talk.")
        if memory.state:
            result.headers = ["key", "value"]
            for key, value in memory.state.items():
                result.row(key, str(value)[:preview])
        result.line("The bounded history and state are projected into context on the next turn.")
        result.line("They are dropped when the conversation ends; nothing here is long-term.")
        return result


class DreamCommand:
    """Consolidate the dream inbox now, and show what changed (`/dream`).

    The platform owns the command; the agent owns the wiring. The host
    hands over a factory, because a dream writes through the agent's own
    governed memory — its policy, its redactor — which only the agent's
    composition root knows how to build.
    """

    group = "Inspect"
    name = "dream"
    summary = "Consolidate recent conversations into long-term memory"
    usage = "/dream"

    def __init__(self, service: "Callable[[], ConsolidationService]") -> None:
        self._service = service

    async def run(self, args: list[str]) -> CommandResult:
        from agentic_commons.memory.consolidation import change_tree_lines

        report = await self._service().dream()
        result = CommandResult(title=f"Dream — {report.events} turn events from {report.sessions} sessions")
        result.lines = change_tree_lines(report)
        result.line("Use /memory to see everything memory now holds.")
        return result
