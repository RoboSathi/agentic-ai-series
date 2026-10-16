"""The source as a model may query it: in memory, read from Parquet, then sealed.

Each export's Parquet files are loaded into an in-memory DuckDB, one table per
curated view; then external access is disabled and the configuration locked,
so nothing a query says can reach a file, a setting or the network. When the
manifest says a newer export exists, the next call loads it into a fresh
database (a sealed one cannot read files any more).

Every query goes through the `QueryGuard` first and runs under a deadline;
results are bounded by rows and by bytes, and say when they were cut.
"""

import difflib
import json
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
from pydantic import BaseModel

from agent_platform.analytics.catalog import Catalog, CuratedView, load_catalog
from agent_platform.analytics.exporter import SourceManifest, read_manifest
from agent_platform.analytics.metric_sql import MetricDefinition, literal, period_sql
from agent_platform.analytics.query_guard import QueryGuard, QueryRejectedError
from agent_platform.config import get_settings
from agent_platform.logging import get_logger

_SEAL = ("SET enable_external_access = false", "SET lock_configuration = true")
# What DuckDB says when a query names something wrong or compares the wrong types: the model
# gets it back with the columns it could have meant, as a rejection it can correct.
_CORRECTABLE = (
    duckdb.BinderException,
    duckdb.CatalogException,
    duckdb.ParserException,
    duckdb.ConversionException,
)
# A name DuckDB quotes in its error ("Referenced column \"agnet_id\" not found").
_QUOTED = re.compile(r'"([^"]+)"')


class ViewSummary(BaseModel):
    """One view as `list_views` describes it."""

    name: str
    description: str
    rows: int
    available: bool


class ColumnInfo(BaseModel):
    """One column: its type in the source, and what it means."""

    name: str
    type: str
    meaning: str


class ViewDescription(BaseModel):
    """One view as `describe_view` describes it."""

    name: str
    description: str
    columns: list[ColumnInfo]
    sample_rows: list[dict[str, Any]]


class MetricSummary(BaseModel):
    """One metric as `list_metrics` describes it."""

    name: str
    label: str
    hint: str
    group: str
    tier: int
    owner: str
    description: str
    better: str
    null_when: str
    view: str
    dimensions: list[str]
    slo_target: float | None


class QueryResult(BaseModel):
    """A query's answer, bounded."""

    sql: str
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    truncated: bool
    # What to check when the answer may not be what was asked: no rows, or rows cut.
    notes: list[str] = []


class SourceDatabase:
    """The sealed, in-memory source a model queries (Facade over DuckDB + guard)."""

    def __init__(self, parquet_dir: Path | None = None, catalog: Catalog | None = None) -> None:
        self._app_config = get_settings()
        self._parquet_dir = parquet_dir or get_settings().analytics.resolved_parquet_dir
        self._catalog = catalog or load_catalog()
        self._lock = threading.Lock()
        self._connection: duckdb.DuckDBPyConnection | None = None
        self._manifest: SourceManifest | None = None
        self._loaded: set[str] = set()
        self._logger = get_logger("analytics")

    @property
    def manifest(self) -> SourceManifest | None:
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
                f"SELECT * FROM {view.name} LIMIT {self._app_config.analytics.query.sample_rows}"
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

    def agents(self) -> list[str]:
        """Every agent id the source holds data for, across the views that carry one."""
        column = self._app_config.analytics.query.agent_column
        with self._lock:
            connection = self._fresh()
            views = [
                v.name
                for v in self._catalog.views
                if v.name in self._loaded and column in v.columns
            ]
            found = {
                str(row[0])
                for view in views
                for row in connection.execute(f"SELECT DISTINCT {column} FROM {view}").fetchall()
                if row[0] is not None
            }
        return sorted(found)

    def metrics(self) -> list[MetricSummary]:
        """The metric catalogue, with each metric's SLO target."""
        targets = self._app_config.observability.slo_targets
        return [
            MetricSummary(
                **metric.model_dump(
                    include={"name", "label", "hint", "group", "tier", "owner", "description"}
                ),
                better=metric.better,
                null_when=metric.null_when,
                view=metric.view,
                dimensions=list(metric.by),
                slo_target=targets.get(metric.name),
            )
            for metric in self._catalog.metrics
        ]

    def compute(
        self,
        name: str,
        agent_id: str | None = None,
        days: int | None = None,
        by: str | None = None,
    ) -> QueryResult:
        """One catalogue metric, by its own formula, over runtime traffic in the period."""
        metric = self._metric(name, by)
        self._known_agent(agent_id)
        with self._lock:
            connection = self._fresh()
            self._available(metric.view)
            sql = period_sql(
                metric,
                lambda view: view,
                self._app_config.evals.channel,
                days,
                by,
                one_agent=agent_id is not None,
            )
            self._logger.info("Analytics metric", metric=name, agent_id=agent_id, days=days, by=by)
            params = [agent_id] if agent_id is not None else []
            # Bound as a parameter; shown with its value, so the SQL can be read and rerun.
            shown = sql.replace("?", literal(agent_id), 1) if agent_id is not None else sql
            return self._bounded(shown, self._run(connection, sql, params))

    def _known_agent(self, agent_id: str | None) -> None:
        """An agent id the source holds no data for is refused with the ones it does hold,
        so the model corrects the id instead of reading an empty result as "no data"."""
        if agent_id is None:
            return
        known = self.agents()
        if known and agent_id not in known:
            words = self._app_config.analytics.query.rejections
            self._logger.warning("Analytics metric refused", agent_id=agent_id, agents=known)
            raise QueryRejectedError(
                words.unknown_agent.format(agent=agent_id, agents=", ".join(known))
            )

    def _metric(self, name: str, by: str | None) -> MetricDefinition:
        words = self._app_config.analytics.query.rejections
        metric = self._catalog.metric(name)
        if metric is None:
            raise QueryRejectedError(words.unknown_metric.format(name=name))
        if by is not None and by not in metric.by:
            dimensions = ", ".join(metric.by) or "none"
            raise QueryRejectedError(
                words.unknown_dimension.format(name=name, by=by, dimensions=dimensions)
            )
        return metric

    def query(self, sql: str) -> QueryResult:
        """Run one guarded query under the configured deadline and size limits."""
        with self._lock:
            connection = self._fresh()
            guard = QueryGuard(self._loaded)
            try:
                bounded = guard.check(sql)
                self._logger.info("Analytics query", sql=sql)
                cursor = self._ran(connection, bounded, sql, guard)
            except QueryRejectedError as exc:
                self._logger.warning("Analytics query rejected", sql=sql, reason=str(exc))
                raise
            return self._bounded(sql, cursor)

    def _ran(
        self, connection: duckdb.DuckDBPyConnection, bounded: str, sql: str, guard: QueryGuard
    ) -> list[Any]:
        """The guarded query's rows; a wrong name or type, refused with what it could be."""
        try:
            return self._run(connection, bounded)
        except _CORRECTABLE as exc:
            raise QueryRejectedError(self._guidance(str(exc), guard.views_in(sql))) from None

    def _guidance(self, error: str, views: list[str]) -> str:
        """DuckDB's error, each view's real columns, and the closest to any name it quoted."""
        words = self._app_config.analytics.query.rejections
        columns = {
            view: list(found.columns) for view in views if (found := self._catalog.get(view))
        }
        known = [name for names in columns.values() for name in names]
        listed = "; ".join(f"{view}({', '.join(names)})" for view, names in columns.items())
        close = [
            match
            for name in _QUOTED.findall(error)
            for match in difflib.get_close_matches(name, known, n=1)
            if match != name
        ]
        hint = words.did_you_mean.format(names=", ".join(dict.fromkeys(close))) if close else ""
        return words.not_runnable.format(error=error.strip(), columns=listed or "-", hint=hint)

    def _run(
        self, connection: duckdb.DuckDBPyConnection, sql: str, params: list[Any] | None = None
    ) -> list[Any]:
        settings = self._app_config.analytics.query
        timer = threading.Timer(settings.timeout_seconds, connection.interrupt)
        timer.start()
        try:
            cursor = connection.execute(sql, params or [])
            columns = [column[0] for column in cursor.description or []]
            return [columns, cursor.fetchall()]
        except duckdb.InterruptException:
            raise QueryRejectedError(
                settings.timed_out.format(seconds=settings.timeout_seconds)
            ) from None
        finally:
            timer.cancel()

    def _bounded(self, sql: str, result: list[Any]) -> QueryResult:
        settings = self._app_config.analytics.query
        columns, rows = result
        truncated = len(rows) > settings.max_rows
        rows = [list(row) for row in rows[: settings.max_rows]]
        while rows and len(_json(rows)) > settings.max_result_bytes:
            rows, truncated = rows[: len(rows) // 2], True
        return QueryResult(
            sql=sql, columns=columns, rows=json.loads(_json(rows)), row_count=len(rows),
            truncated=truncated, notes=_notes(len(rows), truncated),
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
                self._app_config.analytics.query.rejections.unknown_view.format(name=name)
            )
        return view

    def _fresh(self) -> duckdb.DuckDBPyConnection:
        """The loaded database, reloaded first if a newer export has been written."""
        manifest = read_manifest(self._parquet_dir)
        stale = self._connection is None or (
            manifest is not None
            and (self._manifest is None or manifest.exported_at != self._manifest.exported_at)
        )
        if stale:
            self._load(manifest)
        assert self._connection is not None
        return self._connection

    def _load(self, manifest: SourceManifest | None) -> None:
        connection = duckdb.connect(":memory:")
        loaded = set()
        for view in self._catalog.views:
            path = self._parquet_dir / f"{view.name}.parquet"
            if path.exists():
                literal = "'" + str(path).replace("'", "''") + "'"
                connection.execute(
                    f"CREATE TABLE {view.name} AS SELECT * FROM read_parquet({literal})"
                )
                loaded.add(view.name)
        timezone = self._app_config.analytics.query.timezone.replace("'", "''")
        connection.execute(f"SET TimeZone = '{timezone}'")
        for statement in _SEAL:
            connection.execute(statement)
        if self._connection is not None:
            self._connection.close()
        self._connection, self._manifest, self._loaded = connection, manifest, loaded
        self._logger.info("Analytics source loaded", views=sorted(loaded))


def _notes(rows: int, truncated: bool) -> list[str]:
    """What the reader should check before trusting the answer: nothing came back, or not all."""
    words = get_settings().analytics.query.notes
    if not rows:
        return [words.empty]
    return [words.truncated.format(rows=rows)] if truncated else []


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
    "SourceDatabase",
    "MetricSummary",
    "QueryResult",
    "ViewDescription",
    "ViewSummary",
]
