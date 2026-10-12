"""Reading the audit trail — one renderer for `make audit <agent>`, `/audit` and the console.

An agent's decisions are recorded in two ledgers: the shared one (DuckDB)
a running agent writes to, and the agent's own (SQLite) that offline
work — a dream, a terminal review — writes to so it never contends with a
running console. `AuditTrail` reads both and merges them by time, so
"what happened" has one answer.

The trail holds tokens, never a value a token stands for: personal data
was tokenised before anything was recorded, and the only way back is
`make pii-reveal`, which is itself an audited act.
"""

import re
from collections.abc import Sequence
from datetime import datetime, timedelta

from pydantic import BaseModel

from agent_platform.audit.ledger import AuditReader
from agent_platform.audit.types import STANDING_FIELD, AuditEntry, Decision
from agent_platform.config import get_settings
from agent_platform.interface.presentation.text_table import Align, box_table
from agent_platform.interface.presentation.text_tree import TreeNode, tree_lines
from agent_platform.observability.types import utc_now


class AuditQuery(BaseModel):
    """What to read: newest first, narrowed by any of these."""

    limit: int
    decision: Decision | None = None
    standing: str | None = None
    # The start of an action's name (`review.`, `support.process_refund`).
    kind: str | None = None
    since: datetime | None = None


class AuditTrail:
    """Every ledger an agent writes to, read as one."""

    def __init__(self, readers: Sequence[AuditReader]) -> None:
        self._readers = readers

    def entries(self, query: AuditQuery) -> list[AuditEntry]:
        """The newest matching entries across every ledger, newest first."""
        found = [
            entry
            for reader in self._readers
            for entry in reader.recent(query.limit, query.decision, query.standing)
            if (query.kind is None or entry.action.startswith(query.kind))
            and (query.since is None or entry.timestamp >= query.since)
        ]
        found.sort(key=lambda entry: entry.timestamp, reverse=True)
        return found[: query.limit]

    def trace(self, trace_id: str) -> list[AuditEntry]:
        """One trace's decisions from every ledger, in the order they were made."""
        found = [entry for reader in self._readers for entry in reader.reconstruct(trace_id)]
        return sorted(found, key=lambda entry: entry.timestamp)


def since(text: str) -> datetime:
    """`24h`, `7d`, or an ISO date or time, as the moment the window starts."""
    view = get_settings().audit.view
    match = re.fullmatch(view.since_pattern, text)
    if match is None:
        moment = datetime.fromisoformat(text)
        return moment if moment.tzinfo else moment.astimezone()
    unit = view.since_units[match.group("unit")]
    return utc_now() - timedelta(**{unit: int(match.group("n"))})


def standing_of(entry: AuditEntry) -> str:
    return str(entry.authority_context.get(STANDING_FIELD) or "")


def row(entry: AuditEntry) -> list[str]:
    """One entry as a table row: when, actor, action, decision, standing, trace, why."""
    words = get_settings().commands.audit
    unknown, limit = words.unknown, words.rationale_chars
    why = entry.rationale if len(entry.rationale) <= limit else f"{entry.rationale[:limit]}…"
    return [
        entry.timestamp.astimezone().strftime(words.time_format),
        entry.actor,
        entry.action or unknown,
        entry.decision,
        standing_of(entry) or unknown,
        entry.trace_id or unknown,
        why or unknown,
    ]


def colour(entry: AuditEntry) -> str:
    """The colour an entry's outcome is drawn in: its standing's, else its decision's."""
    view = get_settings().audit.view
    return view.standing_colours.get(standing_of(entry)) or view.decision_colours.get(
        entry.decision, ""
    )


def table_lines(entries: Sequence[AuditEntry]) -> list[str]:
    """The entries as a box table, as plain lines (the console colours by state itself)."""
    words = get_settings().commands.audit
    align: list[Align] = ["left"] * len(words.headers)
    return box_table(words.headers, [row(entry) for entry in entries], align)


def trace_tree_lines(trace_id: str, entries: Sequence[AuditEntry]) -> list[str]:
    """One trace as a tree: each decision, then the details that explain it."""
    view, words = get_settings().audit.view, get_settings().commands.audit
    root = TreeNode(trace_id)
    for entry in entries:
        node = root.add(
            view.trace_entry.format(
                time=entry.timestamp.astimezone().strftime(words.time_format),
                action=entry.action or words.unknown,
                decision=entry.decision,
                actor=entry.actor,
            )
        )
        if entry.rationale:
            node.add(entry.rationale)
        for key in view.trace_details:
            value = entry.authority_context.get(key)
            if value not in (None, "", [], {}):
                node.add(view.trace_detail.format(key=key, value=value))
    return tree_lines(root)[1:]
