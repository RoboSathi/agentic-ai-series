"""`/dlq` — review the dead-letter queue: the messages a security screen quarantined.

Every agent's screens (prompt injection, content moderation) put what they block in the same
platform queue, so the command lives here and works for any agent. It lists entries by status,
shows one entry with everything recorded about it, and lets a reviewer claim, resolve or discard
one. A review never re-runs the quarantined message, and every action is written to the audit
trail first. The same lines print from `make dlq <agent>` (`dlq_cli.py`).

    /dlq [pending|in_review|resolved|discarded|all] [limit]
    /dlq <id>
    /dlq claim <id>
    /dlq resolve <id> <note…>
    /dlq discard <id> <note…>
"""

import getpass
from typing import Any, get_args

from agent_platform.audit import AuditEntry, SqliteAuditLedger
from agent_platform.commands._pool import PoolBackedCommand
from agent_platform.commands.types import CommandError, CommandResult
from agent_platform.config import get_settings
from agent_platform.logging import new_trace_id
from agent_platform.presentation.text_table import Align, box_table
from agent_platform.presentation.text_tree import value_lines
from agent_platform.resilience.dead_letter_queue.queue import (
    DeadLetterEntry,
    DeadLetterQueue,
    DeadLetterStatus,
)

STATUSES: tuple[str, ...] = get_args(DeadLetterStatus)
ALL = "all"
CLAIM, RESOLVE, DISCARD = "claim", "resolve", "discard"
ACTOR = "dlq_review"
# The id column right-aligned, the rest left.
_ALIGN: list[Align] = ["right", "left", "left", "left", "left", "left", "left"]


class DeadLetterCommand(PoolBackedCommand):
    """List, show and review quarantined messages."""

    group = "Inspect"
    name = "dlq"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._words = get_settings().commands.dlq
        self.summary = self._words.summary
        self.usage = self._words.usage

    async def run(self, args: list[str]) -> CommandResult:
        if args and args[0] in (CLAIM, RESOLVE, DISCARD):
            return self._act(args[0], args[1:])
        if len(args) == 1 and args[0].isdigit():
            return self._show(int(args[0]))
        status = args[0] if args and not args[0].isdigit() else self._words.default_status
        if status != ALL and status not in STATUSES:
            raise CommandError(self._words.unknown_status.format(status=status, usage=self.usage))
        return self._list(status, _limit(args))

    # -- reading -------------------------------------------------------------------------

    def _list(self, status: str, limit: int) -> CommandResult:
        queue = self._queue()
        entries = queue.entries(None if status == ALL else status, limit)  # type: ignore[arg-type]
        counts = queue.counts()
        result = CommandResult(title=self._words.list_title.format(status=status, n=len(entries)))
        result.line(
            self._words.counts.format(
                counts=" · ".join(f"{s} {counts.get(s, 0)}" for s in STATUSES)
            )
        )
        if not entries:
            return result.line(self._words.empty)
        rows = [self._row(entry) for entry in entries]
        for line in box_table(self._words.headers, rows, _ALIGN):
            result.line(line)
        return result.line(self._words.show_hint)

    def _row(self, entry: DeadLetterEntry) -> list[str]:
        preview = self._words.preview_chars
        text = entry.payload if len(entry.payload) <= preview else f"{entry.payload[:preview]}…"
        decided = ", ".join(entry.metadata.get("decided_by", [])) or self._words.unknown
        return [
            str(entry.id),
            entry.created_at.astimezone().strftime(self._words.time_format),
            str(entry.metadata.get("screen", entry.reason)),
            entry.status,
            entry.session_id or self._words.unknown,
            decided,
            text,
        ]

    def _show(self, entry_id: int) -> CommandResult:
        entry = self._entry(entry_id)
        result = CommandResult(title=self._words.entry_title.format(id=entry.id))
        fields = entry.model_dump(mode="json")
        for line in value_lines(fields):
            result.line(line)
        return result

    # -- reviewing -----------------------------------------------------------------------

    def _act(self, action: str, args: list[str]) -> CommandResult:
        if not args or not args[0].isdigit():
            raise CommandError(self._words.needs_id.format(action=action, usage=self.usage))
        entry_id, note = int(args[0]), " ".join(args[1:]).strip()
        if action != CLAIM and not note:
            raise CommandError(self._words.needs_note.format(action=action))
        self._entry(entry_id)  # an unknown id is an error before anything is written
        reviewer = getpass.getuser()
        self._audit(action, entry_id, reviewer, note)
        queue = self._queue()
        done = (
            queue.mark_in_review(entry_id, reviewer)
            if action == CLAIM
            else queue.resolve(entry_id, reviewer, note, discarded=action == DISCARD)
        )
        if not done:
            status = self._entry(entry_id).status
            raise CommandError(
                self._words.refused.format(action=action, id=entry_id, status=status)
            )
        return CommandResult(
            title=self._words.done.format(action=action, id=entry_id, reviewer=reviewer)
        )

    def _audit(self, action: str, entry_id: int, reviewer: str, note: str) -> None:
        """Written before the change: an attempted review is on the record either way."""
        SqliteAuditLedger(self.pool, agent_id=get_settings().agent.id).record(
            AuditEntry(
                trace_id=new_trace_id(),
                actor=ACTOR,
                action=self._words.audit_action.format(action=action),
                authority_context={"entry_id": entry_id, "reviewer": reviewer},
                decision="allow",
                rationale=note or action,
            )
        )

    def _entry(self, entry_id: int) -> DeadLetterEntry:
        entry = self._queue().get(entry_id)
        if entry is None:
            raise CommandError(self._words.unknown_id.format(id=entry_id))
        return entry

    def _queue(self) -> DeadLetterQueue:
        return DeadLetterQueue(self.pool, agent_id=get_settings().agent.id)


def _limit(args: list[str]) -> int:
    numbers = [int(arg) for arg in args[1:] if arg.isdigit()]
    return numbers[0] if numbers else get_settings().commands.default_limit


__all__ = ["DeadLetterCommand"]
