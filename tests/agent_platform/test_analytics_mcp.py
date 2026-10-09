"""The analytics MCP server: guarded, sealed, bounded SQL over the source, served over MCP.

The source is built by the real exporter from real ledgers; the MCP client connects
to the server in-process (the SDK's memory transport), so the protocol is real too.
"""

from pathlib import Path
from typing import Any

import duckdb
import pytest
from mcp import Client

from agent_platform.analytics import ParquetExporter
from agent_platform.analytics.source_db import SourceDatabase
from agent_platform.analytics.mcp_server import build_server
from agent_platform.analytics.query_guard import QueryGuard, QueryRejectedError
from agent_platform.config import get_settings
from agent_platform.logging.trace_context import bound_log_context
from agent_platform.observability.facts import DuckDbFactLedger, LlmCallFact
from agent_platform.observability.store import DuckDbObservabilityStore

VIEWS = ["llm_calls", "turns"]


@pytest.fixture
def parquet(tmp_path: Path) -> Path:
    store = DuckDbObservabilityStore(tmp_path / "observability.duckdb")
    facts = DuckDbFactLedger(store, agent_id="agent-a")
    with bound_log_context(agent="agent-a", session="s", trace_id="t"):
        for index in range(5):
            facts.record(LlmCallFact(model=f"m{index % 2}", latency_ms=100.0 * index))
    directory = tmp_path / "parquet"
    ParquetExporter(store, parquet_dir=directory, evals_db=tmp_path / "none.duckdb").export()
    return directory


# -- the guard ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT model, count(*) FROM llm_calls GROUP BY 1",
        "WITH slow AS (SELECT * FROM llm_calls WHERE latency_ms > 100) SELECT * FROM slow",
        "SELECT * FROM llm_calls UNION ALL SELECT * FROM llm_calls",
        "SELECT t.trace_id FROM turns t JOIN llm_calls c USING (trace_id)",
    ],
)
def test_a_select_over_the_views_is_allowed_and_bounded(sql: str) -> None:
    bounded = QueryGuard(VIEWS).check(sql)
    assert bounded.endswith(f"LIMIT {get_settings().analytics.query.max_rows + 1}")


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1; DROP TABLE turns",
        "DELETE FROM turns",
        "INSERT INTO turns VALUES (1)",
        "CREATE TABLE x AS SELECT 1",
        "COPY (SELECT 1) TO '/tmp/x.csv'",
        "ATTACH '/tmp/x.duckdb' AS x",
        "PRAGMA version",
        "SET enable_external_access = true",
        "SELECT * FROM read_parquet('/etc/hosts')",
        "SELECT * FROM read_csv_auto('/etc/hosts')",
        "SELECT * FROM duckdb_settings()",
        "SELECT * FROM main.turns",
        "SELECT * FROM platform_audit_entries",
        "SELECT getenv('HOME')",
        "SELEC nonsense",
    ],
)
def test_anything_else_is_refused_with_a_reason(sql: str) -> None:
    with pytest.raises(QueryRejectedError) as refused:
        QueryGuard(VIEWS).check(sql)
    assert str(refused.value)


# -- the sealed database ------------------------------------------------------------------


def test_views_list_what_was_exported(parquet: Path) -> None:
    views = {view.name: view for view in SourceDatabase(parquet).views()}
    assert views["llm_calls"].rows == 5 and views["llm_calls"].available
    assert not views["eval_runs"].available  # no eval store was attached


def test_describe_gives_types_meanings_and_samples(parquet: Path) -> None:
    description = SourceDatabase(parquet).describe("llm_calls")
    columns = {column.name: column for column in description.columns}
    assert columns["latency_ms"].type == "DOUBLE" and columns["latency_ms"].meaning
    assert len(description.sample_rows) == get_settings().analytics.query.sample_rows


def test_describing_an_unknown_view_is_refused(parquet: Path) -> None:
    with pytest.raises(QueryRejectedError):
        SourceDatabase(parquet).describe("platform_audit_entries")


def test_a_query_answers_with_columns_and_rows(parquet: Path) -> None:
    result = SourceDatabase(parquet).query(
        "SELECT model, count(*) AS calls FROM llm_calls GROUP BY 1 ORDER BY 1"
    )
    assert result.columns == ["model", "calls"]
    assert result.rows == [["m0", 3], ["m1", 2]] and not result.truncated


def test_rows_beyond_the_cap_are_cut_and_said_to_be(
    parquet: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(get_settings().analytics.query, "max_rows", 2)
    result = SourceDatabase(parquet).query("SELECT * FROM llm_calls")
    assert result.row_count == 2 and result.truncated


def test_the_database_cannot_reach_files_even_past_the_guard(parquet: Path) -> None:
    database = SourceDatabase(parquet)
    database.views()  # loads and seals
    connection = database._connection  # the second wall, tested on its own
    assert connection is not None
    with pytest.raises(duckdb.Error):
        connection.execute(f"SELECT * FROM read_parquet('{parquet / 'llm_calls.parquet'}')")
    with pytest.raises(duckdb.Error):
        connection.execute("SET enable_external_access = true")


def test_a_newer_export_is_picked_up(parquet: Path, tmp_path: Path) -> None:
    database = SourceDatabase(parquet)
    assert database.query("SELECT count(*) FROM llm_calls").rows == [[5]]
    store = DuckDbObservabilityStore(tmp_path / "observability.duckdb")
    DuckDbFactLedger(store).record(LlmCallFact(model="m9"))
    ParquetExporter(store, parquet_dir=parquet, evals_db=tmp_path / "none.duckdb").export()
    assert database.query("SELECT count(*) FROM llm_calls").rows == [[6]]


# -- the protocol --------------------------------------------------------------------------


async def _call(parquet: Path, tool: str, arguments: dict[str, Any]) -> Any:
    async with Client(build_server(SourceDatabase(parquet))) as client:
        return await client.call_tool(tool, arguments)


async def test_the_server_offers_three_read_only_tools(parquet: Path) -> None:
    async with Client(build_server(SourceDatabase(parquet))) as client:
        tools = (await client.list_tools()).tools
    assert {tool.name for tool in tools} == {"list_views", "describe_view", "run_query"}
    assert all(tool.annotations and tool.annotations.read_only_hint for tool in tools)


async def test_run_query_over_mcp(parquet: Path) -> None:
    result = await _call(parquet, "run_query", {"sql": "SELECT count(*) AS n FROM llm_calls"})
    assert not result.is_error
    assert result.structured_content["rows"] == [[5]]


async def test_a_refused_query_tells_the_model_why(parquet: Path) -> None:
    result = await _call(parquet, "run_query", {"sql": "DROP TABLE llm_calls"})
    assert result.is_error
    assert get_settings().analytics.query.rejections.not_a_select in result.content[0].text


async def test_the_manifest_is_a_resource(parquet: Path) -> None:
    async with Client(build_server(SourceDatabase(parquet))) as client:
        resource = await client.read_resource(get_settings().analytics.mcp_server.manifest_uri)
    assert '"llm_calls"' in resource.contents[0].text
