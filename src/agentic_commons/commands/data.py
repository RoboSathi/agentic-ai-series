"""Commands that read what the platform itself stores (LLD §20).

Memory is a platform subsystem — every agent built on this harness has
the same four CoALA stores — so `/memory` belongs here.

Commands that read a *product's* tables do not. `/users`, `/orders` and
`/queue` used to live in this module, and required one particular
product's database to exist before the console could open at all. They
now belong to the agent that has users and orders, and reach the console
through its `AgentHost`.
"""

from typing import TYPE_CHECKING

from agentic_commons.commands._pool import PoolBackedCommand
from agentic_commons.commands.types import CommandResult
from agentic_commons.config import get_settings
from agentic_commons.persistence import SqliteConnectionPool

if TYPE_CHECKING:
    from agentic_commons.memory import MemoryFacade


class MemoryCommand(PoolBackedCommand):
    """Show what the CoALA stores hold — long-term, or this conversation's.

    Working memory lives in the running process, not the database, so
    the console has to be handed the live facade to show it. Without
    one, `/memory working` says so rather than reporting an empty window
    that would read as "nothing was remembered".
    """

    name = "memory"
    summary = "What memory holds (add 'working' for this conversation)"
    usage = "/memory [working|<search term>]"

    _WORKING = "working"

    def __init__(
        self,
        pool: SqliteConnectionPool | None = None,
        memory: "MemoryFacade | None" = None,
        session_id: str = "",
    ) -> None:
        super().__init__(pool)
        self._memory = memory
        self._session_id = session_id

    async def run(self, args: list[str]) -> CommandResult:
        from agentic_commons.memory import MemoryQuery, SemanticStore

        settings = get_settings().commands
        if args and args[0].casefold() == self._WORKING:
            return self._working_window(settings.value_preview_chars)
        pool = self.pool
        semantic = SemanticStore(pool)
        term = " ".join(args)
        facts = semantic.read(MemoryQuery(text=term, limit=settings.default_limit))

        result = CommandResult(
            title=f"Semantic memory — {len(facts)} fact(s)"
            + (f" matching '{term}'" if term else "")
        )
        if not facts:
            return result.line("Nothing stored yet.")
        result.headers = ["key", "value", "provenance", "confidence"]
        for fact in facts:
            result.row(
                fact.key,
                fact.value[: settings.value_preview_chars],
                fact.provenance.kind,
                f"{fact.confidence:.2f}",
                style="yellow" if fact.provenance.kind == "self_reported" else "",
            )
        result.line("Self-reported facts are weighted below verified ones at retrieval.")
        result.line("Use /memory working to see what this conversation is holding.")
        return result

    def _working_window(self, preview: int) -> CommandResult:
        """This session's active window — what the next turn will see."""
        if self._memory is None:
            return CommandResult(title="Working memory", ok=False).line(
                "No live agent is attached to this console."
            )
        facts = self._memory.active_window(self._session_id)
        result = CommandResult(
            title=f"Working memory — {len(facts)} fact(s) active in this conversation"
        )
        if not facts:
            return result.line("Nothing established yet. Verify, and it fills as you talk.")
        result.headers = ["category", "key", "value", "source"]
        for fact in facts:
            result.row(
                fact.category or "—",
                fact.key,
                fact.value[:preview],
                fact.provenance.source,
            )
        result.line("These are pinned into the context window on the next turn.")
        result.line("They are dropped when the conversation ends; nothing here is long-term.")
        return result
