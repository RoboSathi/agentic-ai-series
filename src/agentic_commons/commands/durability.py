"""`/threads`, `/resume`, `/checkpoints` — operating on durable runs (LLD §20).

Platform commands over the `DurableAgent` contract: they read the run
registry and the checkpointer, and resume through the agent. An agent
that does not run durably simply leaves them with nothing to say.
"""

from typing import Any

from agentic_commons.commands.types import CommandError, CommandResult, ConsoleEffect
from agentic_commons.config import get_settings
from agentic_commons.durability import (
    RESUMABLE_STATUSES,
    DurableAgent,
    ResumableRunError,
    ResumeRefusedError,
    RunRecord,
)

# Shortened thread ids, as the tables show them. Any unique prefix resumes.
_ID_WIDTH = 18


class _BoundToAgent:
    """Commands that read the agent on screen, re-bound after every rebuild."""

    def __init__(self) -> None:
        self._agent: Any = None

    def bind(self, host: Any, agent: Any) -> None:
        self._agent = agent

    def _durable(self) -> DurableAgent:
        if not isinstance(self._agent, DurableAgent):
            raise CommandError("This agent does not run durably; there are no threads")
        return self._agent

    def _record(self, prefix: str) -> RunRecord:
        record = self._durable().runs.find(prefix)
        if record is None:
            raise CommandError(f"No single thread matches '{prefix}' — see /threads")
        return record


class ThreadsCommand(_BoundToAgent):
    """Runs that need attention, or the most recent ones."""

    name = "threads"
    summary = "Durable runs: resumable, waiting, recent"
    usage = "/threads [all]"

    async def run(self, args: list[str]) -> CommandResult:
        runs = self._durable().runs
        limit = get_settings().commands.default_limit
        show_all = bool(args) and args[0].casefold() == "all"
        records = runs.recent(limit) if show_all else runs.resumable(limit)
        title = "Recent threads" if show_all else "Threads needing attention"
        result = CommandResult(title=f"{title} — {len(records)}")
        if not records:
            return result.line("Nothing to resume. /threads all lists recent runs.")
        result.headers = ["thread", "status", "stopped at", "waiting on", "tries", "query", "error"]
        for record in records:
            result.row(
                record.thread_id[:_ID_WIDTH],
                record.status,
                record.next_node or "—",
                record.waiting_on or "—",
                str(record.attempts),
                record.preview,
                record.error or "—",
                style="yellow" if record.status in RESUMABLE_STATUSES else "",
            )
        return result.line("Resume one with /resume <thread>; inspect with /checkpoints <thread>.")


class ResumeCommand(_BoundToAgent):
    """Continue a failed thread from its last checkpoint."""

    name = "resume"
    summary = "Resume a failed thread from its last checkpoint"
    usage = "/resume <thread>"

    async def run(self, args: list[str]) -> CommandResult:
        if not args:
            raise CommandError(f"Usage: {self.usage}")
        record = self._record(args[0])
        if record.status == "interrupted":
            raise CommandError(
                f"{record.thread_id[:_ID_WIDTH]} is waiting on {record.waiting_on}; "
                "it resumes when that decision is made, not by hand"
            )
        try:
            resumed = await self._durable().resume(record.thread_id)
        except ResumeRefusedError as exc:
            raise CommandError(str(exc)) from exc
        except ResumableRunError as exc:
            return CommandResult(title="Resume failed again", ok=False).line(str(exc))
        result = CommandResult(
            title=f"Resumed {record.thread_id[:_ID_WIDTH]} — {resumed.status}",
            effect=ConsoleEffect(session_id=resumed.session_id),
        )
        return result.line(f"ASSISTANT  {resumed.response}" if resumed.response else "No reply.")


class CheckpointsCommand(_BoundToAgent):
    """One thread's checkpoint history."""

    name = "checkpoints"
    summary = "A thread's checkpoints, node by node"
    usage = "/checkpoints <thread>"

    async def run(self, args: list[str]) -> CommandResult:
        if not args:
            raise CommandError(f"Usage: {self.usage}")
        record = self._record(args[0])
        history = await self._durable().checkpoint_history(record.thread_id)
        result = CommandResult(title=f"Checkpoints — {record.thread_id} ({record.status})")
        result.headers = ["step", "written by", "next", "checkpoint", "at"]
        for item in history:
            result.row(
                str(item.step),
                ", ".join(item.wrote) or item.source,
                ", ".join(item.next_nodes) or "(end)",
                item.checkpoint_id[-_ID_WIDTH:],
                item.created_at[11:19],
            )
        return result
