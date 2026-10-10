"""The analytics source: curated views exported as Parquet, checked against the semantic layer.

A real DuckDB per test, filled through the platform's own writers (fact store, audit
ledger); the source is read back with DuckDB, as Rill and the MCP server read it.
"""

from pathlib import Path
from typing import Any

import duckdb
import pytest

from agent_platform.analytics import (
    Catalog,
    ParquetExporter,
    ViewContractError,
    load_catalog,
    read_manifest,
)
from agent_platform.analytics.cli import manifest_lines
from agent_platform.audit import AuditEntry, DuckDbAuditLedger
from agent_platform.config import get_settings
from agent_platform.logging.trace_context import bound_log_context
from agent_platform.observability.facts import (
    DuckDbFactStore,
    ModelCallFact,
    ToolCallFact,
    TurnFact,
)
from agent_platform.observability.store import DuckDbObservabilityStore


@pytest.fixture
def store(tmp_path: Path) -> DuckDbObservabilityStore:
    """Two turns in one session: one with a primary and a backup model call, one with none."""
    store = DuckDbObservabilityStore(tmp_path / "observability.duckdb")
    facts = DuckDbFactStore(store, agent_id="agent-a")
    with bound_log_context(agent="agent-a", session="s1", trace_id="t1"):
        first = {"task_id": "k1", "turn_id": "u1"}
        facts.append(ModelCallFact(**first, role="planner", model="m1", total_tokens=10))
        facts.append(
            ModelCallFact(**first, role="planner", model="m2", is_backup=True, total_tokens=5)
        )
        facts.append(TurnFact(**first, status="completed", steps=1, latency_ms=100))
        facts.append(TurnFact(task_id="k2", turn_id="u2", status="completed", steps=0))
    DuckDbAuditLedger(store, agent_id="agent-a").record(
        AuditEntry(
            trace_id="t2",
            actor="agent-a",
            action="support.process_refund",
            authority_context={"standing": "RATIFIABLE", "session_id": "s1", "query": "secret"},
            decision="escalate",
            rationale="over the limit; customer is jane@example.com",
        )
    )
    return store


def _export(store: DuckDbObservabilityStore, tmp_path: Path, **kwargs: Any) -> Path:
    parquet = tmp_path / "parquet"
    ParquetExporter(
        store, parquet_dir=parquet, evals_db=tmp_path / "no-evals.duckdb", **kwargs
    ).export()
    return parquet


def _read(parquet: Path, view: str, sql: str = "SELECT * FROM v") -> list[tuple[Any, ...]]:
    connection = duckdb.connect()
    connection.execute(f"CREATE VIEW v AS SELECT * FROM read_parquet('{parquet / view}.parquet')")
    return connection.execute(sql).fetchall()


def _columns(parquet: Path, view: str) -> list[str]:
    described = duckdb.connect().execute(
        f"DESCRIBE SELECT * FROM read_parquet('{parquet / view}.parquet')"
    )
    return [row[0] for row in described.fetchall()]


def test_every_exported_view_has_exactly_its_documented_columns(
    store: DuckDbObservabilityStore, tmp_path: Path
) -> None:
    parquet = _export(store, tmp_path)
    manifest = read_manifest(parquet)
    assert manifest is not None
    for view in load_catalog().views:
        if manifest.views[view.name].skipped:
            continue
        assert _columns(parquet, view.name) == list(view.columns), view.name


def test_the_manifest_counts_each_view_and_says_why_one_was_skipped(
    store: DuckDbObservabilityStore, tmp_path: Path
) -> None:
    manifest = read_manifest(_export(store, tmp_path))
    assert manifest is not None
    assert manifest.schema_version == get_settings().analytics.schema_version
    assert manifest.views["model_calls"].rows == 2
    assert manifest.views["turns"].rows == 2
    assert manifest.views["eval_runs"].skipped


def test_turns_carry_the_calls_and_tokens_made_in_them(
    store: DuckDbObservabilityStore, tmp_path: Path
) -> None:
    parquet = _export(store, tmp_path)
    rows = _read(parquet, "turns", "SELECT turn_id, model_calls, tokens FROM v ORDER BY 1")
    assert rows == [("u1", 2, 15), ("u2", 0, 0)]


def test_a_turn_counts_its_tool_proposals(store: DuckDbObservabilityStore, tmp_path: Path) -> None:
    DuckDbFactStore(store, agent_id="agent-a").append(
        ToolCallFact(task_id="k1", turn_id="u1", tool="x.y", stage="executed", succeeded=False)
    )
    parquet = _export(store, tmp_path)
    rows = _read(parquet, "turns", "SELECT turn_id, tool_proposals FROM v ORDER BY 1")
    assert rows == [("u1", 1), ("u2", 0)]


def test_a_backup_call_is_marked_as_one(store: DuckDbObservabilityStore, tmp_path: Path) -> None:
    parquet = _export(store, tmp_path)
    assert _read(parquet, "model_calls", "SELECT model, is_backup FROM v ORDER BY 1") == [
        ("m1", False),
        ("m2", True),
    ]


def test_audit_decisions_expose_standing_but_never_the_raw_context(
    store: DuckDbObservabilityStore, tmp_path: Path
) -> None:
    parquet = _export(store, tmp_path)
    (row,) = _read(parquet, "audit_decisions", "SELECT standing, session_id, rationale FROM v")
    assert row[:2] == ("RATIFIABLE", "s1")
    assert "jane@example.com" not in row[2] and "<EMAIL>" in row[2]
    assert "authority_context" not in _columns(parquet, "audit_decisions")


def test_an_undocumented_column_fails_the_export(
    store: DuckDbObservabilityStore, tmp_path: Path
) -> None:
    view = load_catalog().views[0].model_copy(update={"columns": {"only": "one column"}})
    with pytest.raises(ViewContractError):
        _export(store, tmp_path, catalog=Catalog(views=[view]))


def test_an_empty_database_exports_empty_views(tmp_path: Path) -> None:
    store = DuckDbObservabilityStore(tmp_path / "empty.duckdb")
    manifest = read_manifest(_export(store, tmp_path))
    assert manifest is not None and manifest.views["model_calls"].rows == 0


def test_no_partial_files_are_left_behind(store: DuckDbObservabilityStore, tmp_path: Path) -> None:
    parquet = _export(store, tmp_path)
    assert not [path for path in parquet.iterdir() if path.name.endswith(".partial")]


def test_an_empty_per_turn_view_says_it_fills_as_agents_run(tmp_path: Path) -> None:
    store = DuckDbObservabilityStore(tmp_path / "empty.duckdb")
    manifest = read_manifest(_export(store, tmp_path))
    assert manifest is not None
    words = get_settings().analytics.words
    lines = manifest_lines(manifest)
    assert any(line.endswith(f"model_calls · {words.empty_hint}") for line in lines)
    assert lines[-1] == words.no_facts


def test_a_populated_view_shows_its_rows_and_no_hint(
    store: DuckDbObservabilityStore, tmp_path: Path
) -> None:
    manifest = read_manifest(_export(store, tmp_path))
    assert manifest is not None
    words = get_settings().analytics.words
    lines = manifest_lines(manifest)
    assert any(line.endswith("model_calls · " + words.rows.format(rows=2)) for line in lines)
    assert words.no_facts not in lines
