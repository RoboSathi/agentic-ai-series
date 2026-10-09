"""The Rill project, generated from the semantic layer (ADR 0009 D3).

Dashboards are not written by hand, because a hand-written measure is a second
copy of a formula. Each page is a Rill *canvas*:

- **Agent Platform** — every agent together: each catalogue metric over all of
  their traffic, and each section ranking the agents side by side;
- **one dashboard per agent** — the same sections over that agent's traffic,
  then the sections its own semantic layer adds (its Tier-2 metrics).

Each scope (the platform, or one agent) gets one model per base view it reads
(`<scope>__<view>`: the view's Parquet, runtime traffic only, filtered to the
agent) and one metrics view per model, whose measures are the catalogue's own
formulas (`expr`), so every number is exact over whatever range is chosen. A
measure is titled "label (hint)", and a view's count is named for what a row is
("Tasks", "Turns"), never "rows".

A page or section is generated only when the source holds the Parquet it reads, so
an export from an older export yields less, never something broken. Every file
starts with a marker line; regenerating replaces exactly those files.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import yaml

from agent_platform.analytics.catalog import Catalog, CuratedView, load_catalog
from agent_platform.analytics.metric_sql import MetricDefinition, literal
from agent_platform.config import get_settings
from agent_platform.config.paths import REPO_ROOT
from agent_platform.config.settings import (
    DashboardGroup,
    DashboardRanking,
    DashboardSettings,
    DashboardTable,
)

_MODELS, _METRICS, _CANVASES = "models", "metrics", "dashboards"
_PARQUET = ".parquet"
_TASKS_VIEW, _CHECKS_VIEW = "tasks", "checks"
_TIER1 = 1
_NAMESPACE = "{scope}__{view}"
_TIME, _DAY, _CHANNEL, _AGENT = "recorded_at", "day", "channel", "agent_id"
# A column is a dimension when its Parquet type is one of these.
_DIMENSION_TYPES = ("VARCHAR", "BOOLEAN")
# How many of a view's measures a by-agent ranking shows when none are named.
_RANKED_MEASURES = 4
_CANVAS = {"type": "canvas"}
_KPI, _TREND, _RANKING, _TABLE, _TEXT = (
    "kpi_grid",
    "line_chart",
    "leaderboard",
    "table",
    "markdown",
)


@dataclass(frozen=True)
class Scope:
    """Whose traffic a dashboard shows: every agent's (`agent` None), or one agent's."""

    name: str
    title: str
    about: str
    agent: str | None = None


@dataclass
class ScopeViews:
    """The metrics views generated for one scope: names and measures, by base view."""

    names: dict[str, str] = field(default_factory=dict)
    measures: dict[str, list[str]] = field(default_factory=dict)
    # The checks the scope's traffic ran (so a screen the agent does not run shows no tiles).
    checks_run: set[str] = field(default_factory=set)


class DashboardGenerator:
    """Writes the Rill project's models, metrics views and canvases (Builder)."""

    def __init__(self, catalog: Catalog | None = None, parquet_dir: Path | None = None) -> None:
        self._catalog = catalog or load_catalog()
        self._parquet_dir = parquet_dir or get_settings().analytics.resolved_parquet_dir
        self._settings: DashboardSettings = get_settings().analytics.dashboards
        self._eval_channel = get_settings().evals.channel

    def write(self, project: Path | None = None) -> tuple[int, list[str]]:
        """Replace the generated files; (files written, agents with a dashboard)."""
        root = project or self._settings.resolved_project_dir
        self._clear(root)
        agents = self.agents()
        written = 0
        for scope in self._scopes(agents):
            files = self._scope(scope)
            for path, body in files:
                self._file(root / path, body)
            written += len(files)
        return written, agents

    def agents(self) -> list[str]:
        """Agents with tasks in the source, and agents with a registered layer."""
        found = set(self._catalog.agents)
        tasks = self._parquet(_TASKS_VIEW)
        if tasks.exists():
            with duckdb.connect() as connection:
                rows = connection.execute(
                    f"SELECT DISTINCT {_AGENT} FROM read_parquet({literal(str(tasks))})"
                ).fetchall()
            found |= {str(row[0]) for row in rows}
        return sorted(found)

    def models(self) -> list[str]:
        """The Rill models the generated project has (what a refresh must reload)."""
        root = self._settings.resolved_project_dir / _MODELS
        return sorted(path.stem for path in root.glob("*.yaml"))

    # -- scopes -----------------------------------------------------------------------------

    def _scopes(self, agents: list[str]) -> list[Scope]:
        platform = self._settings.platform
        titles = _agent_titles(self._settings.agent_configs_glob)
        agent_scopes = [
            Scope(agent, titles.get(agent) or agent, self._settings.agent_about, agent)
            for agent in agents
        ]
        return [Scope(platform.name, platform.display_name, platform.about), *agent_scopes]

    def _scope(self, scope: Scope) -> list[tuple[Path, dict[str, Any]]]:
        """One scope's models, metrics views and canvas (nothing when it has no data)."""
        metrics = self._metrics(scope)
        groups = self._groups(scope)
        views = [v for v in self._views(metrics, groups) if self._parquet(v.name).exists()]
        if not views:
            return []
        built = ScopeViews(checks_run=self._checks_run(scope))
        files: list[tuple[Path, dict[str, Any]]] = []
        for view in views:
            name = _NAMESPACE.format(scope=scope.name, view=_short(view))
            measures = self._measures(view, metrics)
            built.names[view.name] = name
            built.measures[view.name] = [measure["name"] for measure in measures]
            files.append((Path(_MODELS) / f"{name}.yaml", self._model(view, scope)))
            files.append(
                (Path(_METRICS) / f"{name}.yaml", self._metrics_view(view, name, measures))
            )
        canvas = CanvasBuilder(self._settings, scope, built, groups).build(metrics)
        files.append((Path(_CANVASES) / f"{scope.name}.yaml", canvas))
        return files

    def _metrics(self, scope: Scope) -> list[MetricDefinition]:
        """Tier 1 for every scope; an agent's own (Tier 2) on its own dashboard."""
        return [
            metric
            for metric in self._catalog.metrics
            if metric.tier == _TIER1 or (scope.agent is not None and metric.owner == scope.agent)
        ]

    def _views(
        self, metrics: list[MetricDefinition], groups: dict[str, DashboardGroup]
    ) -> list[CuratedView]:
        """The counted base views the scope's metrics and rankings read, in catalogue order."""
        wanted = {metric.view for metric in metrics}
        for group in groups.values():
            wanted |= {ranking.view for ranking in group.rankings}
            wanted |= {kpis.view for kpis in group.kpis}
            wanted |= {group.table.view} if group.table else set()
        return [view for view in self._catalog.views if view.name in wanted and view.count]

    def _groups(self, scope: Scope) -> dict[str, DashboardGroup]:
        """The platform's sections, then (on an agent's dashboard) the agent's own."""
        owned = {m.group for m in self._catalog.metrics if scope.agent and m.owner == scope.agent}
        own = {name: group for name, group in self._catalog.groups.items() if name in owned}
        return {**self._settings.groups, **own}

    # -- one view: model and metrics view ---------------------------------------------------

    def _model(self, view: CuratedView, scope: Scope) -> dict[str, Any]:
        """The platform's model reads the Parquet (runtime traffic only); an agent's selects its
        rows from the platform's, or reads the agent's own view. Views, not tables: queries
        are fast at this size, and materializing only slowed start-up."""
        shared = scope.agent is not None and not view.owner
        conditions = []
        if _CHANNEL in view.columns and not shared:
            conditions.append(f"{_CHANNEL} <> {literal(self._eval_channel)}")
        if scope.agent is not None and _AGENT in view.columns:
            conditions.append(f"{_AGENT} = {literal(scope.agent)}")
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        source = (
            _NAMESPACE.format(scope=self._settings.platform.name, view=view.name)
            if shared
            else f"read_parquet('{self._settings.data_link}/{view.name}{_PARQUET}')"
        )
        return {
            "version": 1,
            "type": "model",
            "connector": "duckdb",
            "sql": f"SELECT * FROM {source}{where}",
        }

    def _checks_run(self, scope: Scope) -> set[str]:
        """The checks this scope's runtime traffic ran (empty before any were recorded)."""
        path = self._parquet(_CHECKS_VIEW)
        if not path.exists():
            return set()
        sql = (
            f"SELECT DISTINCT check_name FROM read_parquet({literal(str(path))}) "
            f"WHERE {_CHANNEL} <> ?"
        )
        params: list[str] = [self._eval_channel]
        if scope.agent is not None:
            sql += f" AND {_AGENT} = ?"
            params.append(scope.agent)
        with duckdb.connect() as connection:
            return {str(row[0]) for row in connection.execute(sql, params).fetchall()}

    def _metrics_view(
        self, view: CuratedView, name: str, measures: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return {
            "version": 1,
            "type": "metrics_view",
            "display_name": view.count.label if view.count else name,
            "description": view.description,
            "model": name,
            "timeseries": _TIME if _TIME in view.columns else _DAY,
            "dimensions": [
                {
                    "name": column,
                    "display_name": column.replace("_", " ").capitalize(),
                    "column": column,
                }
                for column in self._dimensions(view)
            ],
            "measures": measures,
        }

    def _measures(self, view: CuratedView, metrics: list[MetricDefinition]) -> list[dict[str, Any]]:
        """The view's count (named for what a row is), then every metric computed over it."""
        counted = []
        if view.count is not None:
            counted.append(
                {
                    "name": view.count.name,
                    "display_name": view.count.title,
                    "description": view.description,
                    "expression": "COUNT(*)",
                    "format_preset": self._settings.default_format,
                }
            )
        return counted + [
            {
                "name": metric.name,
                "display_name": metric.title,
                "description": f"{metric.description} Better: {metric.better}.",
                "expression": metric.expr,
                "format_preset": self._format(metric.name),
            }
            for metric in metrics
            if metric.view == view.name
        ]

    def _dimensions(self, view: CuratedView) -> list[str]:
        path = literal(str(self._parquet(view.name)))
        with duckdb.connect() as connection:
            rows = connection.execute(f"DESCRIBE SELECT * FROM read_parquet({path})").fetchall()
        typed = {str(row[0]): str(row[1]) for row in rows}
        return [
            column
            for column in view.columns
            if typed.get(column, "").startswith(_DIMENSION_TYPES) and column != _DAY
        ]

    def _format(self, name: str) -> str:
        return next(
            (preset for ending, preset in self._settings.formats.items() if name.endswith(ending)),
            self._settings.default_format,
        )

    def _parquet(self, view: str) -> Path:
        return self._parquet_dir / f"{view}{_PARQUET}"

    # -- files ------------------------------------------------------------------------------

    def _file(self, path: Path, body: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        text = yaml.dump(body, Dumper=_PlainDumper, sort_keys=False, allow_unicode=True, width=100)
        path.write_text(f"{self._settings.marker}\n{text}", encoding="utf-8")

    def _clear(self, root: Path) -> None:
        """Remove what an earlier run generated (and only that)."""
        for folder in (_MODELS, _METRICS, _CANVASES):
            for path in (root / folder).glob("*.yaml"):
                first = path.read_text(encoding="utf-8").split("\n", 1)[0]
                if first == self._settings.marker:
                    path.unlink()


class CanvasBuilder:
    """One scope's canvas: a title, then a section per group (KPIs, trends, rankings)."""

    def __init__(
        self,
        settings: DashboardSettings,
        scope: Scope,
        views: ScopeViews,
        groups: dict[str, DashboardGroup],
    ) -> None:
        self._settings = settings
        self._layout = settings.layout
        self._scope = scope
        self._views = views
        self._groups = groups
        self._counted: set[str] = set()

    def build(self, metrics: list[MetricDefinition]) -> dict[str, Any]:
        rows = [self._title()]
        for name, group in self._groups.items():
            rows.extend(self._section(group, [m for m in metrics if m.group == name]))
        return {
            **_CANVAS,
            "display_name": self._scope.title,
            "defaults": {"time_range": self._settings.time_range, "comparison_mode": "time"},
            "rows": rows,
        }

    def _title(self) -> dict[str, Any]:
        content = f"# {self._scope.title}\n{self._scope.about}"
        return self._row([self._markdown(content)], self._layout.title_height)

    def _section(
        self, group: DashboardGroup, metrics: list[MetricDefinition]
    ) -> list[dict[str, Any]]:
        """A header, its KPI grids, then its charts, rankings and tables, two to a row."""
        views = _unique([m.view for m in metrics] + [r.view for r in group.rankings])
        views = [view for view in views if view in self._views.names]
        kpis = self._explicit(group) if group.kpis else self._default(views, metrics)
        table = self._table(group.table) if group.table else None
        if not kpis and table is None:
            return []
        header = self._markdown(f"## {group.title}\n{group.about}")
        rows = [self._row([header], self._layout.header_height)]
        if kpis:
            rows.append(self._row(self._spread(kpis), self._layout.kpi_height))
        charts = [*self._trends(group, metrics), *self._rankings(group)]
        charts += [table] if table else []
        charts += self._by_agent(group, kpis)
        for start in range(0, len(charts), 2):
            pair = self._spread(charts[start : start + 2])
            rows.append(self._row(pair, self._layout.chart_height))
        return rows

    def _explicit(self, group: DashboardGroup) -> list[dict[str, Any]]:
        """The section's titled grids, each shown when its view, measures and checks are there."""
        grids = []
        for kpis in group.kpis:
            available = self._views.measures.get(kpis.view, [])
            measures = [measure for measure in kpis.measures if measure in available]
            ran = not kpis.requires or bool(set(kpis.requires) & self._views.checks_run)
            if measures and ran:
                grids.append(self._grid(kpis.view, measures, kpis.title))
        return grids

    def _default(self, views: list[str], metrics: list[MetricDefinition]) -> list[dict[str, Any]]:
        """One grid per view: its count (the first time the view appears), then the metrics."""
        grids = []
        for view in views:
            available = self._views.measures[view]
            measures = [m.name for m in metrics if m.view == view and m.name in available]
            if view not in self._counted and available and available[0] not in measures:
                measures.insert(0, available[0])
            self._counted.add(view)
            if measures:
                grids.append(self._grid(view, measures, None))
        return grids

    def _grid(self, view: str, measures: list[str], title: str | None) -> dict[str, Any]:
        grid: dict[str, Any] = {
            "metrics_view": self._views.names[view],
            "measures": measures,
            "comparison": self._layout.comparison,
            "sparkline": self._layout.sparkline,
        }
        if title:
            grid["title"] = title
        return {_KPI: grid}

    def _table(self, table: DashboardTable) -> dict[str, Any] | None:
        if table.view not in self._views.names:
            return None
        available = self._views.measures[table.view]
        measures = [measure for measure in table.measures if measure in available]
        return {
            _TABLE: {
                "metrics_view": self._views.names[table.view],
                "title": table.title,
                "columns": [*table.by, *measures],
            }
        }

    def _by_agent(self, group: DashboardGroup, kpis: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """On the platform page: the section's first KPI grid, one row per agent."""
        if self._scope.agent is not None or not kpis:
            return []
        first = kpis[0][_KPI]
        by = self._settings.by_agent
        columns = [by.dimension, *first["measures"][: by.measures]]
        title = by.title.format(section=group.title)
        return [
            {_TABLE: {"metrics_view": first["metrics_view"], "title": title, "columns": columns}}
        ]

    def _trends(
        self, group: DashboardGroup, metrics: list[MetricDefinition]
    ) -> list[dict[str, Any]]:
        by_name = {metric.name: metric for metric in metrics}
        trends = [by_name[name] for name in group.trends if name in by_name]
        return [
            {
                _TREND: {
                    "metrics_view": self._views.names[metric.view],
                    "title": metric.title,
                    "x": {"field": _TIME, "type": "temporal"},
                    "y": {"field": metric.name, "type": "quantitative", "zeroBasedOrigin": True},
                }
            }
            for metric in trends
            if metric.view in self._views.names
        ]

    def _rankings(self, group: DashboardGroup) -> list[dict[str, Any]]:
        rankings = [
            self._leaderboard(ranking, None)
            for ranking in group.rankings
            if ranking.view in self._views.names
        ]
        return [ranking for ranking in rankings if ranking]

    def _leaderboard(self, ranking: DashboardRanking, title: str | None) -> dict[str, Any]:
        available = self._views.measures[ranking.view]
        named = [measure for measure in ranking.measures if measure in available]
        measures = named or available[:_RANKED_MEASURES]
        if not measures:
            return {}
        return {
            _RANKING: {
                "metrics_view": self._views.names[ranking.view],
                "title": title or f"By {ranking.by.replace('_', ' ')}",
                "dimensions": [ranking.by],
                "measures": measures,
                "num_rows": self._layout.ranking_rows,
            }
        }

    def _spread(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        width = max(self._layout.columns // max(len(items), 1), 1)
        return [{**item, "width": width} for item in items]

    @staticmethod
    def _row(items: list[dict[str, Any]], height: str) -> dict[str, Any]:
        return {"items": items, "height": height}

    def _markdown(self, content: str) -> dict[str, Any]:
        alignment = {"horizontal": "left", "vertical": "middle"}
        return {
            _TEXT: {"content": content, "alignment": alignment},
            "width": self._layout.columns,
        }


class _PlainDumper(yaml.SafeDumper):
    """YAML with no anchors or aliases: Rill parses each canvas component on its own, so a
    value shared between two components must be written out in both."""

    def ignore_aliases(self, data: Any) -> bool:
        return True


def _short(view: CuratedView) -> str:
    """A view's name within its owner's scope (an agent's own views lose its prefix)."""
    prefix = _NAMESPACE.format(scope=view.owner, view="")
    return view.name.removeprefix(prefix) if view.owner else view.name


def _unique(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _agent_titles(pattern: str) -> dict[str, str]:
    """Each agent's `agent.display_name`, read from its own config (by convention)."""
    titles: dict[str, str] = {}
    for path in sorted(REPO_ROOT.glob(pattern)):
        agent = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("agent") or {}
        if agent.get("id"):
            titles[str(agent["id"])] = str(agent.get("display_name") or "")
    return titles


__all__ = ["CanvasBuilder", "DashboardGenerator", "Scope"]
