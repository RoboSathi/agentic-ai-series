"""The analytics source: every curated view, exported as Parquet beside a manifest.

Readers outside the agent's process (the analytics MCP server, Rill) cannot open
the observability database while an agent holds its lock, and should not read
raw tables anyway. So the agent's own connection renders each curated view and
copies it to `analytics.parquet_dir/<view>.parquet`, written to a temporary name and
renamed into place so a reader never sees half a file. The manifest says when
and what: each view's row count, or why it was skipped.

A view's columns are checked against the semantic layer on every export: a
column added without a meaning, or documented but gone, fails the export rather
than reaching a dashboard that silently reads nothing.

An agent's own (Tier 2) view whose tables do not exist yet — the agent has not
recorded anything — is skipped with its reason, and the metrics over it are
left out of the catalogue views, so one agent's empty layer never breaks the
export for the rest.
"""

import json
import os
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from agent_platform.analytics.catalog import Catalog, CuratedView, load_catalog, render_sql
from agent_platform.audit import DuckDbAuditLedger
from agent_platform.config import get_settings
from agent_platform.config.paths import resolve_path
from agent_platform.logging import get_logger
from agent_platform.observability.facts import TIER1_FACTS, fact_table_ddl
from agent_platform.observability.store import DuckDbObservabilityStore
from agent_platform.observability.types import utc_now

# Each view is staged as a temporary view of this name on the exporting connection.
_STAGED = "mart_{name}"
_PARQUET, _PARTIAL = ".parquet", ".partial"


class ViewContractError(RuntimeError):
    """A view's columns no longer match what the semantic layer says it has."""


class ExportedView(BaseModel):
    """One view as the manifest records it."""

    rows: int = 0
    file: str = ""
    skipped: str = ""


class SourceManifest(BaseModel):
    """What the source holds, and as of when."""

    schema_version: int
    exported_at: datetime = Field(default_factory=utc_now)
    views: dict[str, ExportedView] = Field(default_factory=dict)


class ParquetExporter:
    """Renders the semantic layer over the agent's connection and writes the Parquet export."""

    def __init__(
        self,
        store: DuckDbObservabilityStore,
        catalog: Catalog | None = None,
        parquet_dir: Path | None = None,
        evals_db: Path | None = None,
    ) -> None:
        self._app_config = get_settings()
        settings = get_settings()
        self._store = store
        self._catalog = catalog or load_catalog()
        self._parquet_dir = parquet_dir or settings.analytics.resolved_parquet_dir
        self._evals_db = evals_db or resolve_path(settings.evals.store.db_path)
        self._logger = get_logger("analytics")

    def export(self) -> SourceManifest:
        """Every view, in order; the manifest last, so it never names a file not yet written."""
        self._parquet_dir.mkdir(parents=True, exist_ok=True)
        # Days and windows are UTC calendar days (as the semantic layer documents them),
        # whatever zone the exporting machine is in.
        timezone = self._app_config.analytics.query.timezone.replace("'", "''")
        self._store.execute_script(f"SET TimeZone = '{timezone}'")
        self._ensure_sources()
        manifest = SourceManifest(schema_version=self._app_config.analytics.schema_version)
        evals_reason = self._attach_evals()
        skipped: set[str] = set()
        try:
            for view in self._catalog.views:
                catalog = self._catalog.without_views(skipped)
                entry = (
                    ExportedView(skipped=evals_reason)
                    if view.reads == "evals" and evals_reason
                    else self._export_owned(view, catalog)
                )
                if entry.skipped:
                    skipped.add(view.name)
                manifest.views[view.name] = entry
        finally:
            self._detach_evals()
        self._remove_retired()
        self._write_manifest(manifest)
        self._logger.info(
            "Analytics source exported",
            parquet_dir=str(self._parquet_dir),
            views={name: entry.rows for name, entry in manifest.views.items()},
        )
        return manifest

    def _export_owned(self, view: CuratedView, catalog: Catalog) -> ExportedView:
        """A platform view must build; an agent's may be skipped, saying why."""
        if not view.owner:
            return self._export(view, catalog)
        try:
            return self._export(view, catalog)
        except ViewContractError:
            raise
        except Exception as exc:  # noqa: BLE001 - one agent's empty layer must not stop the source
            self._logger.info("Agent view skipped", view=view.name, reason=str(exc))
            return ExportedView(skipped=str(exc).splitlines()[0])

    def _export(self, view: CuratedView, catalog: Catalog) -> ExportedView:
        staged = _STAGED.format(name=view.name)
        sql = render_sql(view, lambda name: _STAGED.format(name=name), catalog)
        self._store.execute_script(f"CREATE OR REPLACE TEMP VIEW {staged} AS {sql}")
        self._check_contract(view, staged)
        target = self._parquet_dir / f"{view.name}{_PARQUET}"
        partial = target.with_suffix(_PARQUET + _PARTIAL)
        self._store.execute_script(
            f"COPY (SELECT * FROM {staged}) TO {_literal(partial)} (FORMAT parquet)"
        )
        os.replace(partial, target)
        row = self._store.query_one(f"SELECT count(*) AS n FROM {staged}")
        return ExportedView(rows=int(row["n"]) if row else 0, file=target.name)

    def _check_contract(self, view: CuratedView, staged: str) -> None:
        actual = [row["column_name"] for row in self._store.iter_rows(f"DESCRIBE {staged}")]
        if actual != list(view.columns):
            raise ViewContractError(
                f"view '{view.name}' has columns {actual}, the semantic layer documents "
                f"{list(view.columns)}"
            )

    def _ensure_sources(self) -> None:
        """Every table the views read, so a fresh database exports empty views, not errors."""
        for fact in TIER1_FACTS:
            self._store.execute_script(fact_table_ddl(fact))
        DuckDbAuditLedger(self._store)

    def _remove_retired(self) -> None:
        """Parquet left by a view the semantic layer no longer has (an older export's)."""
        current = set(self._catalog.names)
        for path in self._parquet_dir.glob(f"*{_PARQUET}"):
            if path.stem not in current:
                path.unlink()

    def _attach_evals(self) -> str:
        """Attach the eval store read-only; the reason it could not be, or ""."""
        alias = self._app_config.analytics.evals_alias
        if not self._evals_db.exists():
            return f"no eval store at {self._evals_db}"
        try:
            self._store.execute_script(f"DETACH DATABASE IF EXISTS {alias}")
            self._store.execute_script(f"ATTACH {_literal(self._evals_db)} AS {alias} (READ_ONLY)")
        except Exception as exc:  # noqa: BLE001 - an eval run holding the lock skips evals only
            return f"eval store unavailable: {exc}"
        return ""

    def _detach_evals(self) -> None:
        alias = self._app_config.analytics.evals_alias
        self._store.execute_script(f"DETACH DATABASE IF EXISTS {alias}")

    def _write_manifest(self, manifest: SourceManifest) -> None:
        target = self._parquet_dir / self._app_config.analytics.manifest_file
        partial = target.with_name(target.name + _PARTIAL)
        partial.write_text(json.dumps(manifest.model_dump(mode="json"), indent=2), encoding="utf-8")
        os.replace(partial, target)


def read_manifest(parquet_dir: Path | None = None) -> SourceManifest | None:
    """The source's manifest, or None before the first export."""
    directory = parquet_dir or get_settings().analytics.resolved_parquet_dir
    path = directory / get_settings().analytics.manifest_file
    if not path.exists():
        return None
    return SourceManifest.model_validate_json(path.read_text(encoding="utf-8"))


def _literal(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


__all__ = [
    "ExportedView",
    "ViewContractError",
    "ParquetExporter",
    "SourceManifest",
    "read_manifest",
]
