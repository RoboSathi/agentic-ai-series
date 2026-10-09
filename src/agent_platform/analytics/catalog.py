"""The semantic layer as code: each curated view's SQL and columns, and the metric catalogue.

Loaded from `analytics.semantic_layer` (the platform's file) plus every agent
layer found at `analytics.agent_layers_glob` (ADR 0009 D7), rendered to SQL
with its placeholders filled from configuration. The exporter builds the source
from it, the MCP server describes it, the dashboard generator draws from it;
none carries its own copy.

An agent layer names its agent (`agent_id`). Its views are namespaced
`<agent_id>__<view>`, a `{{view:x}}` inside it reads its own view `x` before a
platform one, and its metrics are tier 2, owned by it. The platform never
names an agent: it only finds the files.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from agent_platform.analytics.metric_sql import (
    MetricDefinition,
    catalogue_relation,
    daily_sql,
    literal,
    targets_relation,
    window_sql,
)
from agent_platform.config import get_settings
from agent_platform.config.paths import REPO_ROOT
from agent_platform.config.settings import DashboardGroup, RedactPattern

# Which database a view reads: the observability facts, the eval store, or configuration.
ViewReads = Literal["observability", "evals", "config"]

# How an agent's own view is named in the source.
NAMESPACE = "{agent}__{name}"
# The tier an agent's own metrics belong to.
AGENT_TIER = 2

_VIEW = re.compile(r"\{\{view:(\w+)\}\}")
_REDACT = re.compile(r"\{\{redact\((.+?)\)\}\}")
_EVALS, _EVAL_CHANNEL = "{{evals}}", "{{eval_channel}}"
_TARGETS, _WINDOWS = "{{slo_targets}}", "{{slo_windows}}"
_DIM_METRIC, _DAILY, _WINDOW = "{{dim_metric}}", "{{metric_daily}}", "{{metric_window}}"
_ABANDONED, _ABANDON_HOURS = "{{abandoned}}", "{{abandon_hours}}"


class ViewCount(BaseModel):
    """What one row of a view is, as a dashboard counts it ("Tasks", not "rows")."""

    name: str
    label: str
    hint: str = ""

    @property
    def title(self) -> str:
        return f"{self.label} ({self.hint})" if self.hint else self.label


class CuratedView(BaseModel):
    """One view of the source: where it reads from, its SQL, and what each column means."""

    name: str
    reads: ViewReads
    description: str
    sql: str
    columns: dict[str, str]
    # Built from what agents record each turn: empty until an agent has run one.
    per_turn: bool = False
    # The agent whose layer declares it; empty for the platform's own views.
    owner: str = ""
    # Reads the metric catalogue, so it is built after every agent layer's views.
    after_agent_layers: bool = False
    # What a row is, for a dashboard's count measure; None for a view not counted.
    count: ViewCount | None = None


class SemanticLayer(BaseModel):
    """One semantic-layer file: the platform's, or one agent's."""

    agent_id: str = ""
    views: list[CuratedView] = Field(default_factory=list)
    metrics: list[MetricDefinition] = Field(default_factory=list)
    # Dashboard sections of an agent's own (its Tier-2 metrics name one as `group`).
    groups: dict[str, DashboardGroup] = Field(default_factory=dict)


class Catalog(BaseModel):
    """Every curated view in dependency order, and every metric."""

    views: list[CuratedView]
    metrics: list[MetricDefinition] = Field(default_factory=list)
    groups: dict[str, DashboardGroup] = Field(default_factory=dict)

    def get(self, name: str) -> CuratedView | None:
        return next((view for view in self.views if view.name == name), None)

    def metric(self, name: str) -> MetricDefinition | None:
        return next((metric for metric in self.metrics if metric.name == name), None)

    @property
    def names(self) -> list[str]:
        return [view.name for view in self.views]

    def without_views(self, names: set[str]) -> Catalog:
        """The catalogue less some views (ones that could not be built) and the metrics over them."""
        return Catalog(
            views=[view for view in self.views if view.name not in names],
            metrics=[metric for metric in self.metrics if metric.view not in names],
            groups=self.groups,
        )

    @property
    def agents(self) -> list[str]:
        """The agents that registered a layer of their own."""
        return sorted({view.owner for view in self.views if view.owner})


@cache
def _read(path: Path) -> SemanticLayer:
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return SemanticLayer.model_validate(raw)


def load_catalog(path: Path | None = None, agent_layers: list[Path] | None = None) -> Catalog:
    """The platform layer, then each agent layer after it (so agent views may read platform ones)."""
    settings = get_settings().analytics
    platform = _read(path or settings.resolved_semantic_layer)
    layers = [_namespaced(_read(p)) for p in (agent_layers or discover_agent_layers())]
    agent_views = [view for layer in layers for view in layer.views]
    return Catalog(
        views=[
            *(view for view in platform.views if not view.after_agent_layers),
            *agent_views,
            *(view for view in platform.views if view.after_agent_layers),
        ],
        metrics=[*platform.metrics, *(metric for layer in layers for metric in layer.metrics)],
        groups={name: group for layer in layers for name, group in layer.groups.items()},
    )


def discover_agent_layers() -> list[Path]:
    """Every agent's semantic layer, found by convention (`analytics.agent_layers_glob`).

    The platform's own file matches the same convention and is not an agent's.
    """
    settings = get_settings().analytics
    platform = settings.resolved_semantic_layer.resolve()
    found = REPO_ROOT.glob(settings.agent_layers_glob)
    return sorted(path for path in found if path.resolve() != platform)


def _namespaced(layer: SemanticLayer) -> SemanticLayer:
    """An agent's layer as the source holds it: its views prefixed, its metrics owned."""
    agent = layer.agent_id
    if not agent:
        raise ValueError("An agent's semantic layer must name its agent (`agent_id`)")
    own = {view.name for view in layer.views}

    def name(view: str) -> str:
        return NAMESPACE.format(agent=agent, name=view) if view in own else view

    def sql(text: str) -> str:
        return _VIEW.sub(lambda match: "{{view:" + name(match.group(1)) + "}}", text)

    views = [
        view.model_copy(update={"name": name(view.name), "sql": sql(view.sql), "owner": agent})
        for view in layer.views
    ]
    metrics = [
        metric.model_copy(update={"view": name(metric.view), "tier": AGENT_TIER, "owner": agent})
        for metric in layer.metrics
    ]
    groups = {
        key: group.model_copy(
            update={
                "rankings": [
                    ranking.model_copy(update={"view": name(ranking.view)})
                    for ranking in group.rankings
                ]
            }
        )
        for key, group in layer.groups.items()
    }
    return layer.model_copy(update={"views": views, "metrics": metrics, "groups": groups})


def render_sql(view: CuratedView, relation: Callable[[str], str], catalog: Catalog) -> str:
    """The view's SQL with every placeholder filled; `relation` names an earlier view."""
    settings = get_settings()
    targets = settings.observability.slo_targets
    eval_channel = settings.evals.channel
    sql = _VIEW.sub(lambda match: relation(match.group(1)), view.sql)
    sql = _REDACT.sub(
        lambda match: _masked(match.group(1), settings.analytics.redact_patterns), sql
    )
    replacements = {
        _EVALS: settings.analytics.evals_alias,
        _EVAL_CHANNEL: eval_channel,
        _ABANDONED: settings.analytics.tasks.reserved.abandoned,
        _ABANDON_HOURS: str(settings.analytics.tasks.abandon_after_hours),
        _TARGETS: targets_relation(catalog.metrics, targets),
        _WINDOWS: _windows_relation(settings.analytics.slo_windows_days),
        _DIM_METRIC: catalogue_relation(catalog.metrics, targets),
        _DAILY: daily_sql(catalog.metrics, relation, eval_channel),
        _WINDOW: window_sql(
            catalog.metrics, relation, eval_channel, settings.analytics.slo_windows_days
        ),
    }
    for placeholder, value in replacements.items():
        if placeholder in sql:
            sql = sql.replace(placeholder, value)
    return sql


def _masked(expression: str, patterns: list[RedactPattern]) -> str:
    """`expression` wrapped in one regexp_replace per configured pattern."""
    for rule in patterns:
        expression = (
            f"regexp_replace({expression}, {literal(rule.pattern)}, "
            f"{literal(rule.replacement)}, 'g')"
        )
    return expression


def _windows_relation(days: list[int]) -> str:
    rows = ", ".join(f"({int(day)})" for day in days)
    return f"(VALUES {rows}) AS slo_windows(days)"


__all__ = [
    "AGENT_TIER",
    "Catalog",
    "CuratedView",
    "MetricDefinition",
    "SemanticLayer",
    "ViewCount",
    "ViewReads",
    "discover_agent_layers",
    "load_catalog",
    "render_sql",
]
