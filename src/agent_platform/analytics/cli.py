"""`python -m agent_platform.analytics <command>` — the source and its dashboards, offline.

- `export` refreshes the source. A running console exports on its own clock
  (`analytics.export_interval_seconds`); this is for when none is running: the
  observability database's single-writer lock means it cannot run beside an
  agent in another process, and says so.
- `dashboards` regenerates the Rill project from the semantic layer.
- `models` prints the Rill models the project has (what a refresh reloads).
- `drop-legacy` drops the fact tables ADR 0009 retired (`analytics.legacy_tables`).
- `dummy` generates the demo world (`analytics.dummy`) into its own DuckDB, then
  exports its Parquet.
"""

import sys
from collections.abc import Callable, Sequence

from agent_platform.analytics.catalog import load_catalog
from agent_platform.analytics.dashboards import DashboardGenerator
from agent_platform.analytics.dummy import generate_world
from agent_platform.analytics.exporter import ExportedView, ParquetExporter, SourceManifest
from agent_platform.config import get_settings
from agent_platform.observability.store import DuckDbObservabilityStore
from agent_platform.presentation.text_tree import TreeNode, tree_lines

EXPORT, DASHBOARDS, MODELS, DROP_LEGACY, DUMMY = (
    "export",
    "dashboards",
    "models",
    "drop-legacy",
    "dummy",
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command; print what it did."""
    words = get_settings().analytics.words
    args = list(argv if argv is not None else sys.argv[1:])
    commands: dict[str, Callable[[], int]] = {
        EXPORT: _export,
        DASHBOARDS: _dashboards,
        MODELS: _models,
        DROP_LEGACY: _drop_legacy,
        DUMMY: _dummy,
    }
    command = commands.get(args[0]) if len(args) == 1 else None
    if command is None:
        print(words.usage)
        return 2
    return command()


def _export() -> int:
    """Export the source and print what it holds, as a tree."""
    store = _store()
    if store is None:
        return 1
    manifest = ParquetExporter(store).export()
    print("\n".join(manifest_lines(manifest)))
    return 0


def _dashboards() -> int:
    generator = DashboardGenerator()
    files, agents = generator.write()
    settings = get_settings().analytics
    print(
        settings.words.dashboards_written.format(
            project=settings.dashboards.project_dir, files=files, agents=", ".join(agents)
        )
    )
    return 0


def _models() -> int:
    print(" ".join(DashboardGenerator().models()))
    return 0


def _drop_legacy() -> int:
    store = _store()
    if store is None:
        return 1
    tables = get_settings().analytics.legacy_tables
    for table in tables:
        store.execute_script(f"DROP TABLE IF EXISTS {table}")
    print(get_settings().analytics.words.dropped.format(tables=", ".join(tables)))
    return 0


def _dummy() -> int:
    """A fresh generated world, and its Parquet: what `SOURCE=dummy` points everything at."""
    settings = get_settings().analytics
    source = settings.sources[settings.dummy.source]
    summary = generate_world()
    print(
        settings.words.dummy_written.format(
            db=source.db_path,
            days=summary.days,
            facts=", ".join(f"{agent} {n}" for agent, n in summary.facts.items()),
            audit=sum(summary.audit.values()),
        )
    )
    store = DuckDbObservabilityStore(source.resolved_db_path)
    try:
        manifest = ParquetExporter(store, parquet_dir=source.resolved_parquet_dir).export()
    finally:
        store.close()
    print("\n".join(manifest_lines(manifest, settings.dummy.source)))
    return 0


def _store() -> DuckDbObservabilityStore | None:
    """The observability database, or None (said) when an agent holds its lock."""
    try:
        return DuckDbObservabilityStore()
    except Exception as exc:  # noqa: BLE001 - the lock error is DuckDB's own type
        print(get_settings().analytics.words.locked.format(error=exc))
        return None


def manifest_lines(manifest: SourceManifest, source: str | None = None) -> list[str]:
    """The manifest as a tree: one branch per view, its rows or why it was skipped."""
    settings = get_settings().analytics
    words = settings.words
    name = source or settings.source
    root = TreeNode(
        words.title.format(
            source=name,
            parquet_dir=settings.sources[name].parquet_dir,
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
