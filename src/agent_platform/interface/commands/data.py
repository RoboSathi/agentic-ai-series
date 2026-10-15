"""Commands that read what the platform itself stores (LLD §20).

Memory is a platform subsystem — every agent built on this harness has
the same four CoALA stores — so `/memory` belongs here.

Commands that read a *product's* tables do not. `/review` used to live
in this module, and required one particular product's database to exist
before the console could open at all. It now belongs to the agent that
has a review queue, and reaches the console through its `AgentHost`.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

from agent_platform.config import get_settings
from agent_platform.core.persistence import SqliteConnectionPool
from agent_platform.interface.commands._pool import PoolBackedCommand
from agent_platform.interface.commands.scope import ConsoleScope
from agent_platform.interface.commands.types import CommandResult

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
    summary = "What memory holds, as tables (a verified customer: working + their episodes)"
    usage = "/memory [working|dream|customer <id>|<search term>]"

    _WORKING = "working"
    _DREAM = "dream"
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
        """Scoped to a verified subject: working + their episodic. Unscoped: every store."""
        settings = get_settings()
        if args and args[0].casefold() == self._WORKING:
            return self._working_window(settings.commands.value_preview_chars)
        if args and args[0].casefold() == self._DREAM:
            return self._dream_events(" ".join(args[1:]) or None)
        subject, term = self._narrowing(args)
        if self._scope is not None and self._scope.scoped:
            return self._scoped(subject, term)
        return self._everything(subject, term)

    def _scoped(self, asked: str | None, term: str | None) -> CommandResult:
        """This conversation's working memory and this subject's episodes — nobody else's.

        Semantic facts and skills are the agent's, common to every subject,
        so a subject's view leaves them out (`make show-memory` has them).
        """
        from agent_platform.memory.inspect import episodic_tables, take_snapshot

        assert self._scope is not None
        commands = get_settings().commands
        verified = self._scope.subject()
        if verified is None:
            return CommandResult(title="Memory", ok=False).line(commands.verify_first)
        if asked is not None and asked != verified:
            return CommandResult(title="Memory", ok=False).line(commands.not_your_records)
        snapshot = take_snapshot(verified, term)
        snapshot.episodic = {k: v for k, v in snapshot.episodic.items() if k == verified}
        result = self._titled()
        self._add_working(result)
        for table in episodic_tables(snapshot, commands.memory_tree_events_per_session):
            result.table(table)
        return result

    def _everything(self, subject: str | None, term: str | None) -> CommandResult:
        """Every store, as `make show-memory` prints it, plus this conversation's window."""
        from agent_platform.memory.inspect import snapshot_tables, take_snapshot

        commands = get_settings().commands
        result = self._titled()
        self._add_working(result)
        snapshot = take_snapshot(subject, term)
        for table in snapshot_tables(snapshot, commands.memory_tree_events_per_session):
            result.table(table)
        return result.line(commands.memory_tables.queue_note)

    def _dream_events(self, term: str | None) -> CommandResult:
        """The dream events alone; a verified console sees only its own subject's."""
        from agent_platform.memory.inspect import inbox_table, take_snapshot

        commands = get_settings().commands
        subject: str | None = None
        if self._scope is not None and self._scope.scoped:
            subject = self._scope.subject()
            if subject is None:
                return CommandResult(title="Memory", ok=False).line(commands.verify_first)
        table = inbox_table(take_snapshot(subject, term))
        return self._titled().table(table).line(commands.memory_tables.dream_note)

    def _titled(self) -> CommandResult:
        words = get_settings().commands.memory_tables
        return CommandResult(title=words.title.format(agent=get_settings().agent.id))

    def _add_working(self, result: CommandResult) -> None:
        """This conversation's working memory, when a live agent is attached."""
        from agent_platform.memory.inspect import working_table

        if self._working_memory is None:
            return
        session = self._scope.session_id if self._scope is not None else self._session_id
        result.table(working_table(self._working_memory.get(session)))

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
        from agent_platform.interface.host.memory import build_dream

        return build_dream()
