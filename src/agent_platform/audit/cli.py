"""`make audit <agent> [DECISION=…] [STANDING=…] [KIND=…] [SINCE=…] [N=…] [TRACE=…]`.

The audit trail in a terminal, coloured by outcome: the same rows `/audit`
shows, from every ledger the agent writes to. `TRACE=<id>` prints one
turn's decisions as a tree. Only tokens are shown; what a token stood for
comes from `make pii-reveal`, which is audited.
"""

import argparse
from collections.abc import Sequence

import duckdb
from rich.console import Console
from rich.table import Table

from agent_platform.audit import AuditReader, DuckDbAuditLedger, SqliteAuditLedger
from agent_platform.audit.types import Decision
from agent_platform.audit.view import (
    AuditQuery,
    AuditTrail,
    colour,
    row,
    since,
    trace_tree_lines,
)
from agent_platform.config import get_settings
from agent_platform.observability.store import DuckDbObservabilityStore


def readers(console: Console) -> list[AuditReader]:
    """Both ledgers; only the agent's own when a running console holds the shared one."""
    own: list[AuditReader] = [SqliteAuditLedger()]
    try:
        return [DuckDbAuditLedger(DuckDbObservabilityStore()), *own]
    except duckdb.IOException:
        console.print(get_settings().audit.view.locked, style="yellow")
        return own


def table(entries: Sequence[object], query: AuditQuery) -> Table:
    """The entries as a Rich table, each row in its outcome's colour."""
    words, view = get_settings().commands.audit, get_settings().audit.view
    narrowed = " · ".join(
        str(value) for value in (query.decision, query.standing, query.kind) if value
    )
    title = view.title.format(
        agent=get_settings().agent.id,
        n=len(entries),
        narrowed=words.narrowed_by.format(filter=narrowed) if narrowed else "",
    )
    rich_table = Table(title=title, title_justify="left")
    for header in words.headers:
        rich_table.add_column(header, overflow="fold")
    for entry in entries:
        rich_table.add_row(*row(entry), style=colour(entry))  # type: ignore[arg-type]
    return rich_table


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--decision", choices=["allow", "deny", "escalate"])
    parser.add_argument("--standing")
    parser.add_argument("--kind", help="the start of an action's name")
    parser.add_argument("--since", help="24h, 7d, or a date")
    parser.add_argument("--limit", type=int, default=get_settings().commands.recent_window)
    parser.add_argument("--trace", help="one trace id, as a tree")
    args = parser.parse_args(argv)
    console = Console(highlight=False)
    trail = AuditTrail(readers(console))
    if args.trace:
        entries = trail.trace(args.trace)
        view = get_settings().audit.view
        title = view.trace_title.format(
            agent=get_settings().agent.id, trace_id=args.trace, n=len(entries)
        )
        console.print(title, style="bold")
        for line in trace_tree_lines(args.trace, entries):
            console.print(line, markup=False, soft_wrap=True)
        return
    decision: Decision | None = args.decision
    query = AuditQuery(
        limit=args.limit,
        decision=decision,
        standing=args.standing,
        kind=args.kind,
        since=since(args.since) if args.since else None,
    )
    entries = trail.entries(query)
    if not entries:
        console.print(get_settings().commands.audit.empty)
        return
    console.print(table(entries, query))


if __name__ == "__main__":
    main()
