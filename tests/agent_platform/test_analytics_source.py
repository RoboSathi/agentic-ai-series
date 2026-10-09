"""The analytics source: curated views exported as Parquet, checked against the semantic layer.

A real DuckDB per test, filled through the platform's own writers (fact ledger,
audit ledger); the source is read back with DuckDB, as Rill and the MCP server read it.
"""

from pathlib import Path
from typing import Any

import duckdb
import pytest

from agent_platform.analytics import (
    Catalog,
    ViewContractError,
    ParquetExporter,
    load_catalog,
    read_manifest,
)
from agent_platform.analytics.cli import manifest_lines
from agent_platform.audit import AuditEntry, DuckDbAuditLedger
from agent_platform.config import get_settings
from agent_platform.logging.trace_context import bound_log_context
from agent_platform.observability.facts import (
    DuckDbFactLedger,
    LlmCallFact,
    ToolCallFact,
    TurnOutcomeFact,
)
from agent_platform.observability.store import DuckDbObservabilityStore


@pytest.fixture
def store(tmp_path: Path) -> DuckDbObservabilityStore:
    store = DuckDbObservabilityStore(tmp_path / "observability.duckdb")
    facts = DuckDbFactLedger(store, agent_id="agent-a")
    with bound_log_context(agent="agent-a", session="s1", trace_id="t1"):
        facts.record(LlmCallFact(role="planner", model="m1", total_tokens=10, latency_ms=100))
        facts.record(LlmCallFact(role="planner", model="m2", position=1, total_tokens=5))
        facts.record(TurnOutcomeFact(outcome="answered", answered=True, escalated=False))
    with bound_log_context(agent="agent-a", session="s1", trace_id="t2"):
        facts.record(
            TurnOutcomeFact(outcome="escalated", answered=False, escalated=True, tool_calls=0)
        )
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
    ParquetExporter(store, parquet_dir=parquet, evals_db=tmp_path / "no-evals.duckdb", **kwargs).export()
    return parquet


def _read(parquet: Path, view: str, sql: str = "SELECT * FROM v") -> list[tuple[Any, ...]]:
    connection = duckdb.connect()
    connection.execute(f"CREATE VIEW v AS SELECT * FROM read_parquet('{parquet / view}.parquet')")
    return connection.execute(sql).fetchall()


def test_every_view_is_exported_with_its_documented_columns(
    store: DuckDbObservabilityStore, tmp_path: Path
) -> None:
    parquet = _export(store, tmp_path)
    for view in load_catalog().views:
        if view.source == "evals":
            continue
        connection = duckdb.connect()
        described = connection.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{parquet / view.name}.parquet')"
        ).fetchall()
        assert [row[0] for row in described] == list(view.columns), view.name


def test_the_manifest_counts_each_view_and_says_why_evals_were_skipped(
    store: DuckDbObservabilityStore, tmp_path: Path
) -> None:
    manifest = read_manifest(_export(store, tmp_path))
    assert manifest is not None
    assert manifest.schema_version == get_settings().analytics.schema_version
    assert manifest.views["llm_calls"].rows == 2
    assert manifest.views["turns"].rows == 2
    assert "no eval store" in manifest.views["eval_runs"].skipped


def test_turns_carry_the_tokens_of_their_model_calls(
    store: DuckDbObservabilityStore, tmp_path: Path
) -> None:
    parquet = _export(store, tmp_path)
    rows = _read(parquet, "turns", "SELECT trace_id, llm_calls, total_tokens FROM v ORDER BY 1")
    assert rows == [("t1", 2, 15), ("t2", 0, 0)]


def test_a_backup_call_is_marked_as_one(store: DuckDbObservabilityStore, tmp_path: Path) -> None:
    parquet = _export(store, tmp_path)
    assert _read(parquet, "llm_calls", "SELECT model, is_backup FROM v ORDER BY 1") == [
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
    columns = [c[0] for c in duckdb.connect().execute(
        f"DESCRIBE SELECT * FROM read_parquet('{parquet / 'audit_decisions'}.parquet')"
    ).fetchall()]  # fmt: skip
    assert "authority_context" not in columns


def test_slos_are_computed_only_from_what_turns_observed(
    store: DuckDbObservabilityStore, tmp_path: Path
) -> None:
    parquet = _export(store, tmp_path)
    daily = dict(_read(parquet, "slo_daily", "SELECT metric, observed FROM v"))
    assert daily["task_success_rate"] == pytest.approx(0.5)
    assert daily["escalation_rate"] == pytest.approx(0.5)
    assert "tool_failure_rate" not in daily  # no turn made a tool call


def test_the_error_budget_follows_the_registrys_rule(
    store: DuckDbObservabilityStore, tmp_path: Path
) -> None:
    parquet = _export(store, tmp_path)
    budget = {
        (metric, days): (remaining, within)
        for metric, days, remaining, within in _read(
            parquet,
            "slo_error_budget",
            "SELECT metric, window_days, error_budget_remaining, within_target FROM v",
        )
    }
    target = get_settings().observability.slo_targets["escalation_rate"]
    expected = max(0.0, min(1.0, (target - 0.5) / target))
    days = get_settings().analytics.slo_windows_days[0]
    assert budget[("escalation_rate", days)] == (pytest.approx(expected), False)


def test_an_undocumented_column_fails_the_export(
    store: DuckDbObservabilityStore, tmp_path: Path
) -> None:
    catalog = load_catalog()
    view = catalog.views[0].model_copy(update={"columns": {"only": "one column"}})
    with pytest.raises(ViewContractError):
        _export(store, tmp_path, catalog=Catalog(views=[view]))


def test_an_empty_database_exports_empty_views(tmp_path: Path) -> None:
    store = DuckDbObservabilityStore(tmp_path / "empty.duckdb")
    manifest = read_manifest(_export(store, tmp_path))
    assert manifest is not None and manifest.views["llm_calls"].rows == 0


def test_no_partial_files_are_left_behind(store: DuckDbObservabilityStore, tmp_path: Path) -> None:
    parquet = _export(store, tmp_path)
    assert not [path for path in parquet.iterdir() if path.name.endswith(".partial")]


def test_an_empty_per_turn_view_says_it_fills_as_agents_run(tmp_path: Path) -> None:
    store = DuckDbObservabilityStore(tmp_path / "empty.duckdb")
    manifest = read_manifest(_export(store, tmp_path))
    assert manifest is not None
    words = get_settings().analytics.words
    lines = manifest_lines(manifest)
    assert any(line.endswith(f"llm_calls · {words.empty_hint}") for line in lines)
    assert any(line.endswith("slo_targets · " + words.rows.format(rows=8)) for line in lines)
    assert lines[-1] == words.no_facts


def test_a_populated_view_shows_its_rows_and_no_hint(
    store: DuckDbObservabilityStore, tmp_path: Path
) -> None:
    manifest = read_manifest(_export(store, tmp_path))
    assert manifest is not None
    words = get_settings().analytics.words
    lines = manifest_lines(manifest)
    assert any(line.endswith("llm_calls · " + words.rows.format(rows=2)) for line in lines)
    assert words.no_facts not in lines


def test_a_turn_carries_its_model_tool_and_decision_counts(
    store: DuckDbObservabilityStore, tmp_path: Path
) -> None:
    with bound_log_context(agent="agent-a", session="s1", trace_id="t1"):
        DuckDbFactLedger(store, agent_id="agent-a").record(
            ToolCallFact(action="x.y", succeeded=False)
        )
    parquet = _export(store, tmp_path)
    rows = _read(
        parquet,
        "turns",
        "SELECT trace_id, primary_model, llm_calls, backup_calls, tool_calls_executed, "
        "tool_failures_executed, actions_escalated FROM v ORDER BY 1",
    )
    assert rows == [("t1", "m1", 2, 1, 1, 1, 0), ("t2", "", 0, 0, 0, 0, 1)]


def test_events_hold_every_model_call_tool_call_and_decision(
    store: DuckDbObservabilityStore, tmp_path: Path
) -> None:
    parquet = _export(store, tmp_path)
    kinds = dict(_read(parquet, "events", "SELECT kind, count(*) FROM v GROUP BY 1"))
    assert kinds == {"model_call": 2, "decision": 1}
    (decision,) = _read(
        parquet, "events", "SELECT name, outcome, standing, failed FROM v WHERE kind = 'decision'"
    )
    assert decision == ("support.process_refund", "escalate", "RATIFIABLE", False)
