"""The Rill dashboards read only what the source exports.

Rill itself is not under test (it is a binary; `make dashboard` validates the
project at start-up). What is: every model reads a view the semantic layer
defines, and every column a metrics view names — dimension, time axis, or
inside a measure — is one that view documents. A renamed column then fails
here, not as a dashboard that silently draws nothing.
"""

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from agent_platform.analytics import load_catalog
from agent_platform.config import REPO_ROOT

PROJECT = REPO_ROOT / "dashboards" / "rill"
_PARQUET = re.compile(r"read_parquet\('data/(\w+)\.parquet'\)")
_QUOTED = re.compile(r"'[^']*'")
_IDENTIFIER = re.compile(r"\b([a-z_][a-z0-9_]*)\b(?!\s*\()")
_SQL_WORDS = {"case", "when", "then", "else", "end", "as", "and", "or", "not", "null", "double",
              "distinct", "is", "in", "where", "nullif"}  # fmt: skip


def _files(kind: str) -> list[Path]:
    return sorted((PROJECT / kind).glob("*.yaml"))


def _load(path: Path) -> dict[str, Any]:
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")))


def _model_views() -> dict[str, str]:
    """Each Rill model's name → the source view it reads."""
    views = {}
    for path in _files("models"):
        match = _PARQUET.search(_load(path)["sql"])
        assert match, f"{path.name} does not read a source view"
        views[path.stem] = match.group(1)
    return views


def test_every_model_reads_a_view_the_semantic_layer_defines() -> None:
    catalog = load_catalog()
    for model, view in _model_views().items():
        assert catalog.get(view) is not None, f"model {model} reads unknown view {view}"


@pytest.mark.parametrize("path", _files("metrics"), ids=lambda path: path.stem)
def test_every_column_a_metrics_view_names_is_documented(path: Path) -> None:
    spec = _load(path)
    view = load_catalog().get(_model_views()[spec["model"]])
    assert view is not None
    columns = set(view.columns)
    named = {spec["timeseries"], *(dimension["column"] for dimension in spec["dimensions"])}
    for measure in spec["measures"]:
        plain = _QUOTED.sub("", measure["expression"]).lower()
        named |= {word for word in _IDENTIFIER.findall(plain) if word not in _SQL_WORDS}
    assert named <= columns, f"{path.name} names {sorted(named - columns)}"


@pytest.mark.parametrize("path", _files("dashboards"), ids=lambda path: path.stem)
def test_every_dashboard_explores_a_metrics_view_that_exists(path: Path) -> None:
    metrics = {metrics_path.stem for metrics_path in _files("metrics")}
    assert _load(path)["metrics_view"] in metrics
