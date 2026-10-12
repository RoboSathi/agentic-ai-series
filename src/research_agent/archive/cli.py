"""`make runs ra`: list archived research runs, or print one report.

Read straight from the artifact store — the three frozen artifacts each
run leaves behind — so it works with no agent built and no memory stack
open, and it re-verifies every artifact's hash as it reads it.
"""

import argparse
import sys
from collections.abc import Sequence

from rich.console import Console
from rich.table import Table

from agent_platform.config import get_settings
from agent_platform.knowledge.artifacts import DirectoryArtifactStore
from research_agent.archive.store import REPORT_ARTIFACT
from research_agent.config import get_agent_settings
from research_agent.domain.report import Report
from research_agent.domain.scope import ScopeBrief
from research_agent.rubric.freeze import BRIEF_ARTIFACT

console = Console()
_OK, _MISSING = 0, 1


def run_rows(store: DirectoryArtifactStore) -> list[list[str]]:
    """One row per archived run, newest last, with whether its artifacts still verify."""
    rows = []
    for run_id in store.runs():
        report, brief = store.read(run_id, REPORT_ARTIFACT), store.read(run_id, BRIEF_ARTIFACT)
        delivered = Report.model_validate(report.payload)
        scope = ScopeBrief.model_validate(brief.payload)
        intact = report.verify() and brief.verify()
        unmet = ", ".join(limit.item_id for limit in delivered.limits)
        rows.append([run_id, delivered.status, scope.archetype, scope.topic, unmet, str(intact)])
    return rows


def _list(store: DirectoryArtifactStore) -> int:
    archive = get_agent_settings().archive
    rows = run_rows(store)
    if not rows:
        console.print(archive.list_empty)
        return _OK
    table = Table()
    for header in archive.list_headers:
        table.add_column(header, overflow="fold")
    for row in rows:
        table.add_row(*row)
    console.print(table)
    return _OK


def _show(store: DirectoryArtifactStore, run_id: str) -> int:
    rendered = store.rendered(run_id, REPORT_ARTIFACT)
    if rendered is None:
        console.print(get_agent_settings().messages.unknown_run.format(run_id=run_id))
        return _MISSING
    console.print(rendered)
    return _OK


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m research_agent.archive.cli")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list archived runs")
    show = commands.add_parser("show", help="print one run's report")
    show.add_argument("run_id")
    args = parser.parse_args(argv)
    store = DirectoryArtifactStore(get_settings().artifacts.resolved_root)
    return _list(store) if args.command == "list" else _show(store, args.run_id)


if __name__ == "__main__":
    sys.exit(main())
