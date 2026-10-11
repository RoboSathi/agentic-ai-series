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

from agent_platform.commands._pool import PoolBackedCommand
from agent_platform.commands.scope import ConsoleScope
from agent_platform.commands.types import CommandResult
from agent_platform.config import get_settings
from agent_platform.persistence import SqliteConnectionPool
from agent_platform.presentation.text_tree import tree_lines

if TYPE_CHECKING:
    from agent_platform.memory import WorkingMemoryStore
    from agent_platform.memory.dream import Dream


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
        self._scope: ConsoleScope | None = None

    def bind_scope(self, scope: ConsoleScope) -> None:
        """Follow the console's session, and show only its verified subject's memory."""
        self._scope = scope

    async def run(self, args: list[str]) -> CommandResult:
        """The same tree `make show-memory` prints, narrowed by customer or search term."""
        settings = get_settings()
        if args and args[0].casefold() == self._WORKING:
            return self._working_window(settings.commands.value_preview_chars)
        subject, term = self._narrowing(args)
        if self._scope is not None and self._scope.scoped:
            return self._scoped_tree(subject, term)
        return self._tree(subject, term)

    def _scoped_tree(self, asked: str | None, term: str | None) -> CommandResult:
        """Only the verified subject's memory; anyone else's is refused, not hidden silently."""
        assert self._scope is not None
        settings = get_settings().commands
        verified = self._scope.subject()
        if verified is None:
            return CommandResult(title="Memory", ok=False).line(settings.verify_first)
        if asked is not None and asked != verified:
            return CommandResult(title="Memory", ok=False).line(settings.not_your_records)
        return self._tree(verified, term)

    def _tree(self, subject: str | None, term: str | None) -> CommandResult:
        from agent_platform.memory.inspect import snapshot_tree, take_snapshot
        from agent_platform.persistence import as_platform_store

        settings = get_settings()
        snapshot = take_snapshot(as_platform_store(self.pool), subject, term)
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
        session = self._scope.session_id if self._scope is not None else self._session_id
        memory = self._working_memory.get(session)
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
    """Consolidate this agent's pending dream events now, and show what changed (`/dream`).

    Every agent has it: the dream is built by the platform from the agent's
    configured memory and its host's `MemoryProfile` (or the default). A
    test or an agent may hand in its own factory.
    """

    group = "Inspect"
    name = "dream"
    summary = "Consolidate pending dream events into long-term memory"
    usage = "/dream"

    def __init__(self, dream: "Callable[[], Dream] | None" = None) -> None:
        self._dream = dream

    async def run(self, args: list[str]) -> CommandResult:
        from agent_platform.memory.dream import change_tree_lines

        report = await self._build().dream()
        result = CommandResult(
            title=f"Dream — {report.events} events from {report.sessions} sessions"
        )
        result.lines = change_tree_lines(report)
        result.line("Use /memory to see everything memory now holds.")
        return result

    def _build(self) -> "Dream":
        if self._dream is not None:
            return self._dream()
        from agent_platform.host.memory import build_dream

        return build_dream()
