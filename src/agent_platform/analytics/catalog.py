"""The semantic layer as code: each curated view's SQL, columns and meaning.

Loaded from `analytics.semantic_layer` (one YAML file), rendered to SQL with its
placeholders filled from configuration. The exporter builds the mart from it,
the MCP server describes the mart from it; neither carries its own copy.
"""

import re
from collections.abc import Callable, Mapping
from functools import cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from agent_platform.config import get_settings
from agent_platform.config.settings import RedactPattern
from agent_platform.observability.slo import LOWER_IS_BETTER

ViewSource = Literal["observability", "evals", "config"]

_VIEW = re.compile(r"\{\{view:(\w+)\}\}")
_REDACT = re.compile(r"\{\{redact\((.+?)\)\}\}")
_EVALS, _TARGETS, _WINDOWS = "{{evals}}", "{{slo_targets}}", "{{slo_windows}}"


class CuratedView(BaseModel):
    """One view of the mart: where it reads from, its SQL, and what each column means."""

    name: str
    source: ViewSource
    description: str
    sql: str
    columns: dict[str, str]
    # Built from what agents record each turn: empty until an agent has run one.
    per_turn: bool = False


class Catalog(BaseModel):
    """Every curated view, in dependency order."""

    views: list[CuratedView]

    def get(self, name: str) -> CuratedView | None:
        return next((view for view in self.views if view.name == name), None)

    @property
    def names(self) -> list[str]:
        return [view.name for view in self.views]


@cache
def _load(path: Path) -> Catalog:
    return Catalog.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def load_catalog(path: Path | None = None) -> Catalog:
    """The semantic layer, read once per file."""
    return _load(path or get_settings().analytics.resolved_semantic_layer)


def render_sql(view: CuratedView, relation: Callable[[str], str]) -> str:
    """The view's SQL with every placeholder filled; `relation` names an earlier view."""
    settings = get_settings()
    sql = _VIEW.sub(lambda match: relation(match.group(1)), view.sql)
    sql = _REDACT.sub(
        lambda match: _masked(match.group(1), settings.analytics.redact_patterns), sql
    )
    return (
        sql.replace(_EVALS, settings.analytics.evals_alias)
        .replace(_TARGETS, _targets_relation(settings.observability.slo_targets))
        .replace(_WINDOWS, _windows_relation(settings.analytics.slo_windows_days))
    )


def _masked(expression: str, patterns: list[RedactPattern]) -> str:
    """`expression` wrapped in one regexp_replace per configured pattern."""
    for rule in patterns:
        expression = (
            f"regexp_replace({expression}, {_literal(rule.pattern)}, "
            f"{_literal(rule.replacement)}, 'g')"
        )
    return expression


def _targets_relation(targets: Mapping[str, float]) -> str:
    rows = ", ".join(
        f"({_literal(metric)}, CAST({target} AS DOUBLE), {metric in LOWER_IS_BETTER})"
        for metric, target in targets.items()
    )
    return f"(VALUES {rows}) AS slo_targets(metric, target, lower_is_better)"


def _windows_relation(days: list[int]) -> str:
    rows = ", ".join(f"({int(day)})" for day in days)
    return f"(VALUES {rows}) AS slo_windows(days)"


def _literal(text: str) -> str:
    """A SQL string literal."""
    return "'" + text.replace("'", "''") + "'"


__all__ = ["Catalog", "CuratedView", "ViewSource", "load_catalog", "render_sql"]
