"""Catalogue metrics as SQL: one definition, rendered per day and per SLO window.

A metric is one aggregate (`expr`) over one curated view, reported per agent and
optionally per breakdown (`by`). This module renders that definition into the
relations the source exports — `metric_daily`, `metric_window`, `dim_metric`,
`slo_targets` — so the Fleet dashboard, the MCP server's `compute_metric` and
`/stats` all read one formula, never a copy of it. Runtime metrics exclude eval
traffic (ADR 0009 D5).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, Field

Better = Literal["higher", "lower", "watched"]
# How a metric's rows are grouped in time: by calendar day, or over a window of days.
_DAILY_GRAIN = "CAST(recorded_at AS DATE) AS day"
_WINDOW_GRAIN = "{days} AS window_days"
_DAILY_KEYS = "agent_id, CAST(recorded_at AS DATE)"
_WINDOW_FILTER = "recorded_at >= now() - to_days({days})"
_EMPTY_DAILY = (
    "SELECT CAST(NULL AS VARCHAR) AS agent_id, CAST(NULL AS DATE) AS day, "
    "CAST(NULL AS VARCHAR) AS metric, CAST(NULL AS VARCHAR) AS dimension, "
    "CAST(NULL AS VARCHAR) AS dimension_value, CAST(NULL AS DOUBLE) AS value WHERE false"
)
_EMPTY_WINDOW = _EMPTY_DAILY.replace(
    "CAST(NULL AS DATE) AS day", "CAST(NULL AS INTEGER) AS window_days"
)


class MetricDefinition(BaseModel):
    """One metric of the catalogue (`dim_metric`)."""

    name: str
    view: str
    expr: str
    better: Better
    description: str
    # How a dashboard names it ("label (hint)") and the section it appears in.
    label: str = ""
    hint: str = ""
    group: str = ""
    null_when: str = ""
    by: list[str] = Field(default_factory=list)
    tier: int = 1
    owner: str = ""

    @property
    def lower_is_better(self) -> bool:
        return self.better == "lower"

    @property
    def title(self) -> str:
        """The metric as a reader sees it: its label, with what it means in brackets."""
        label = self.label or self.name.replace("_", " ").capitalize()
        return f"{label} ({self.hint})" if self.hint else label


def daily_sql(
    metrics: Sequence[MetricDefinition], relation: Callable[[str], str], eval_channel: str
) -> str:
    """Every metric per agent per day, overall and per breakdown."""
    parts = [
        _select(metric, relation, eval_channel, _DAILY_GRAIN, _DAILY_KEYS, "") for metric in metrics
    ]
    return _union(parts, _EMPTY_DAILY)


def window_sql(
    metrics: Sequence[MetricDefinition],
    relation: Callable[[str], str],
    eval_channel: str,
    windows: Sequence[int],
) -> str:
    """Every metric per agent over each window of days back from now."""
    parts = [
        _select(
            metric,
            relation,
            eval_channel,
            _WINDOW_GRAIN.format(days=int(days)),
            "agent_id",
            _WINDOW_FILTER.format(days=int(days)),
        )
        for metric in metrics
        for days in windows
    ]
    return _union(parts, _EMPTY_WINDOW)


def catalogue_relation(metrics: Sequence[MetricDefinition], targets: Mapping[str, float]) -> str:
    """The catalogue as rows: `dim_metric`."""
    if not metrics:
        return (
            "(SELECT CAST(NULL AS VARCHAR) AS metric, CAST(NULL AS VARCHAR) AS label, "
            "CAST(NULL AS VARCHAR) AS hint, CAST(NULL AS VARCHAR) AS section, "
            "CAST(NULL AS INTEGER) AS tier, "
            "CAST(NULL AS VARCHAR) AS owner, CAST(NULL AS VARCHAR) AS description, "
            "CAST(NULL AS VARCHAR) AS better, CAST(NULL AS VARCHAR) AS null_when, "
            "CAST(NULL AS VARCHAR) AS view, CAST(NULL AS VARCHAR) AS dimensions, "
            "CAST(NULL AS DOUBLE) AS slo_target WHERE false)"
        )
    rows = ", ".join(
        "("
        + ", ".join(
            [
                literal(metric.name),
                literal(metric.label),
                literal(metric.hint),
                literal(metric.group),
                f"CAST({int(metric.tier)} AS INTEGER)",
                literal(metric.owner),
                literal(metric.description),
                literal(metric.better),
                literal(metric.null_when),
                literal(metric.view),
                literal(",".join(metric.by)),
                _number(targets.get(metric.name)),
            ]
        )
        + ")"
        for metric in metrics
    )
    columns = (
        "metric, label, hint, section, tier, owner, description, better, null_when, view, "
        "dimensions, slo_target"
    )
    return f"(VALUES {rows}) AS dim_metric({columns})"


def targets_relation(metrics: Sequence[MetricDefinition], targets: Mapping[str, float]) -> str:
    """(metric, target, lower_is_better) for each catalogue metric with an SLO target."""
    rows = [
        f"({literal(m.name)}, CAST({targets[m.name]} AS DOUBLE), {m.lower_is_better})"
        for m in metrics
        if m.name in targets
    ]
    if not rows:
        return (
            "(SELECT CAST(NULL AS VARCHAR) AS metric, CAST(NULL AS DOUBLE) AS target, "
            "CAST(NULL AS BOOLEAN) AS lower_is_better WHERE false)"
        )
    return f"(VALUES {', '.join(rows)}) AS slo_targets(metric, target, lower_is_better)"


def period_sql(
    metric: MetricDefinition,
    relation: Callable[[str], str],
    eval_channel: str,
    days: int | None,
    by: str | None,
    one_agent: bool,
) -> str:
    """One metric over one period (all time, or the last `days`), per agent [and breakdown].

    With `one_agent`, the query takes the agent's id as its single parameter.
    """
    conditions = [f"channel <> {literal(eval_channel)}"]
    if days is not None:
        conditions.append(_WINDOW_FILTER.format(days=int(days)))
    if one_agent:
        conditions.append("agent_id = ?")
    keys = ["agent_id", *([by] if by else [])]
    return (
        f"SELECT {', '.join(keys)}, CAST({metric.expr} AS DOUBLE) AS {metric.name} "
        f"FROM {relation(metric.view)} WHERE {' AND '.join(conditions)} "
        f"GROUP BY {', '.join(keys)} ORDER BY {', '.join(keys)}"
    )


def _select(
    metric: MetricDefinition,
    relation: Callable[[str], str],
    eval_channel: str,
    grain: str,
    keys: str,
    window: str,
) -> str:
    """One metric's rows: overall, then once per breakdown."""
    where = f"channel <> {literal(eval_channel)}" + (f" AND {window}" if window else "")
    source = relation(metric.view)
    name = literal(metric.name)
    value = f"CAST({metric.expr} AS DOUBLE) AS value"
    overall = (
        f"SELECT agent_id, {grain}, {name} AS metric, '' AS dimension, '' AS dimension_value, "
        f"{value} FROM {source} WHERE {where} GROUP BY {keys}"
    )
    per_dimension = [
        f"SELECT agent_id, {grain}, {name} AS metric, {literal(dim)} AS dimension, "
        f"coalesce(CAST({dim} AS VARCHAR), '') AS dimension_value, {value} "
        f"FROM {source} WHERE {where} GROUP BY {keys}, {dim}"
        for dim in metric.by
    ]
    return "\nUNION ALL\n".join([overall, *per_dimension])


def _union(parts: list[str], empty: str) -> str:
    return "\nUNION ALL\n".join(parts) if parts else empty


def _number(value: float | None) -> str:
    return "CAST(NULL AS DOUBLE)" if value is None else f"CAST({value} AS DOUBLE)"


def literal(text: str) -> str:
    """A SQL string literal."""
    return "'" + text.replace("'", "''") + "'"


__all__ = [
    "Better",
    "MetricDefinition",
    "catalogue_relation",
    "daily_sql",
    "literal",
    "period_sql",
    "targets_relation",
    "window_sql",
]
