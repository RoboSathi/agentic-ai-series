"""The mart as a model may query it: in memory, read from Parquet, then sealed.

Each export's Parquet files are loaded into an in-memory DuckDB, one table per
curated view; then external access is disabled and the configuration locked,
so nothing a query says can reach a file, a setting or the network. When the
manifest says a newer export exists, the next call loads it into a fresh
database (a sealed one cannot read files any more).

Every query goes through the `QueryGuard` first and runs under a deadline;
results are bounded by rows and by bytes, and say when they were cut.
"""

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
from pydantic import BaseModel

from agent_platform.analytics.catalog import Catalog, CuratedView, load_catalog
from agent_platform.analytics.exporter import MartManifest, read_manifest
from agent_platform.analytics.query_guard import QueryGuard, QueryRejectedError
from agent_platform.config import get_settings
from agent_platform.logging import get_logger

_SEAL = ("SET enable_external_access = false", "SET lock_configuration = true")


class ViewSummary(BaseModel):
    """One view as `list_views` describes it."""

    name: str
    description: str
    rows: int
    available: bool


class ColumnInfo(BaseModel):
    """One column: its type in the mart, and what it means."""

    name: str
    type: str
    meaning: str


class ViewDescription(BaseModel):
    """One view as `describe_view` describes it."""

    name: str
    description: str
    columns: list[ColumnInfo]
    sample_rows: list[dict[str, Any]]


class QueryResult(BaseModel):
    """A query's answer, bounded."""

    sql: str
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    truncated: bool


class MartDatabase:
    """The sealed, in-memory mart a model queries (Facade over DuckDB + guard)."""

    def __init__(self, mart_dir: Path | None = None, catalog: Catalog | None = None) -> None:
        self._mart_dir = mart_dir or get_settings().analytics.resolved_mart_dir
        self._catalog = catalog or load_catalog()
        self._lock = threading.Lock()
        self._connection: duckdb.DuckDBPyConnection | None = None
        self._manifest: MartManifest | None = None
        self._loaded: set[str] = set()
        self._logger = get_logger("analytics")

    @property
    def manifest(self) -> MartManifest | None:
        """The export currently loaded (after refreshing to the newest)."""
        with self._lock:
            self._fresh()
            return self._manifest

    def views(self) -> list[ViewSummary]:
        with self._lock:
            self._fresh()
            return [self._summary(view) for view in self._catalog.views]

    def describe(self, name: str) -> ViewDescription:
        with self._lock:
            connection = self._fresh()
            view = self._available(name)
            types = {
                row[0]: row[1] for row in connection.execute(f"DESCRIBE {view.name}").fetchall()
            }
            sample = connection.execute(
                f"SELECT * FROM {view.name} LIMIT {get_settings().analytics.query.sample_rows}"
            )
            return ViewDescription(
                name=view.name,
                description=view.description,
                columns=[
                    ColumnInfo(name=column, type=types.get(column, ""), meaning=meaning)
                    for column, meaning in view.columns.items()
                ],
                sample_rows=_records(sample),
            )

    def query(self, sql: str) -> QueryResult:
        """Run one guarded query under the configured deadline and size limits."""
        with self._lock:
            connection = self._fresh()
            bounded = QueryGuard(self._loaded).check(sql)
            self._logger.info("Analytics query", sql=sql)
            cursor = self._run(connection, bounded)
            return self._bounded(sql, cursor)

    def _run(self, connection: duckdb.DuckDBPyConnection, sql: str) -> list[Any]:
        settings = get_settings().analytics.query
        timer = threading.Timer(settings.timeout_seconds, connection.interrupt)
        timer.start()
        try:
            cursor = connection.execute(sql)
            columns = [column[0] for column in cursor.description or []]
            return [columns, cursor.fetchall()]
        except duckdb.InterruptException:
            raise QueryRejectedError(
                settings.timed_out.format(seconds=settings.timeout_seconds)
            ) from None
        finally:
            timer.cancel()

    def _bounded(self, sql: str, result: list[Any]) -> QueryResult:
        settings = get_settings().analytics.query
        columns, rows = result
        truncated = len(rows) > settings.max_rows
        rows = [list(row) for row in rows[: settings.max_rows]]
        while rows and len(_json(rows)) > settings.max_result_bytes:
            rows, truncated = rows[: len(rows) // 2], True
        return QueryResult(
            sql=sql, columns=columns, rows=json.loads(_json(rows)), row_count=len(rows),
            truncated=truncated,
        )  # fmt: skip

    def _summary(self, view: CuratedView) -> ViewSummary:
        entry = self._manifest.views.get(view.name) if self._manifest else None
        return ViewSummary(
            name=view.name,
            description=view.description,
            rows=entry.rows if entry else 0,
            available=view.name in self._loaded,
        )

    def _available(self, name: str) -> CuratedView:
        view = self._catalog.get(name)
        if view is None or name not in self._loaded:
            raise QueryRejectedError(
                get_settings().analytics.query.rejections.unknown_view.format(name=name)
            )
        return view

    def _fresh(self) -> duckdb.DuckDBPyConnection:
        """The loaded database, reloaded first if a newer export has been written."""
        manifest = read_manifest(self._mart_dir)
        stale = self._connection is None or (
            manifest is not None
            and (self._manifest is None or manifest.exported_at != self._manifest.exported_at)
        )
        if stale:
            self._load(manifest)
        assert self._connection is not None
        return self._connection

    def _load(self, manifest: MartManifest | None) -> None:
        connection = duckdb.connect(":memory:")
        loaded = set()
        for view in self._catalog.views:
            path = self._mart_dir / f"{view.name}.parquet"
            if path.exists():
                literal = "'" + str(path).replace("'", "''") + "'"
                connection.execute(
                    f"CREATE TABLE {view.name} AS SELECT * FROM read_parquet({literal})"
                )
                loaded.add(view.name)
        timezone = get_settings().analytics.query.timezone.replace("'", "''")
        connection.execute(f"SET TimeZone = '{timezone}'")
        for statement in _SEAL:
            connection.execute(statement)
        if self._connection is not None:
            self._connection.close()
        self._connection, self._manifest, self._loaded = connection, manifest, loaded
        self._logger.info("Analytics mart loaded", views=sorted(loaded))


def _records(cursor: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    columns = [column[0] for column in cursor.description or []]
    rows = [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]
    return list(json.loads(_json(rows)))


def _json(value: Any) -> str:
    """JSON with dates, decimals and the like as their string form."""
    return json.dumps(value, default=_plain)


def _plain(value: Any) -> str:
    return value.isoformat() if isinstance(value, datetime) else str(value)


__all__ = [
    "ColumnInfo",
    "MartDatabase",
    "QueryResult",
    "ViewDescription",
    "ViewSummary",
]
