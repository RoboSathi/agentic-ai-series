"""`python -m agent_platform.analytics export` — refresh the mart outside a console.

A running console exports on its own clock (`analytics.export_interval_seconds`).
This is for when none is running: the observability database's single-writer
lock means it cannot run beside an agent in another process, and says so.
"""

import sys
from collections.abc import Sequence

from agent_platform.analytics.catalog import load_catalog
from agent_platform.analytics.exporter import ExportedView, MartExporter, MartManifest
from agent_platform.config import get_settings
from agent_platform.observability.store import DuckDbObservabilityStore
from agent_platform.presentation.text_tree import TreeNode, tree_lines

EXPORT = "export"


def main(argv: Sequence[str] | None = None) -> int:
    """Export the mart and print what it holds, as a tree."""
    words = get_settings().analytics.words
    args = list(argv if argv is not None else sys.argv[1:])
    if args != [EXPORT]:
        print(words.usage)
        return 2
    try:
        store = DuckDbObservabilityStore()
    except Exception as exc:  # noqa: BLE001 - the lock error is DuckDB's own type
        print(words.locked.format(error=exc))
        return 1
    manifest = MartExporter(store).export()
    print("\n".join(manifest_lines(manifest)))
    return 0


def manifest_lines(manifest: MartManifest) -> list[str]:
    """The manifest as a tree: one branch per view, its rows or why it was skipped."""
    words = get_settings().analytics.words
    root = TreeNode(
        words.title.format(
            mart_dir=get_settings().analytics.resolved_mart_dir,
            exported_at=manifest.exported_at.astimezone().strftime(words.time_format),
        )
    )
    facts = _fact_backed()
    for name, view in manifest.views.items():
        root.add(f"{name} · {_detail(view, name in facts)}")
    lines = tree_lines(root)
    if all(manifest.views[name].rows == 0 for name in facts if name in manifest.views):
        lines.append(words.no_facts)
    return lines


def _detail(view: ExportedView, fact_backed: bool) -> str:
    words = get_settings().analytics.words
    if view.skipped:
        return words.skipped.format(reason=view.skipped)
    if fact_backed and view.rows == 0:
        return words.empty_hint
    return words.rows.format(rows=view.rows)


def _fact_backed() -> set[str]:
    """The views built from what agents record each turn."""
    return {view.name for view in load_catalog().views if view.per_turn}


__all__ = ["main", "manifest_lines"]
