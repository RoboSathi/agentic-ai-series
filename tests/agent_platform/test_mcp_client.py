"""The MCP client: a remote server's tools, arriving as ordinary governed tool definitions.

The server is the platform's real analytics server over a real exported Parquet,
connected in-process through the SDK; only the transport's target is substituted.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_platform.analytics import ParquetExporter
from agent_platform.analytics.source_db import SourceDatabase
from agent_platform.analytics.mcp_server import build_server
from agent_platform.config import get_settings
from agent_platform.config.settings import McpServerSettings, McpToolEnvelope
from agent_platform.mcp import (
    ConfiguredGovernancePolicy,
    GovernedMcpToolSource,
    McpServerSpec,
    McpSessionClosedError,
    SdkMcpTransport,
    server_spec,
)
from agent_platform.mcp.schema import arguments_model
from agent_platform.observability.facts import DuckDbFactLedger, LlmCallFact
from agent_platform.observability.store import DuckDbObservabilityStore
from agent_platform.tools import NamespacedToolRegistry

READ_ONLY = McpToolEnvelope(
    surface="locked",
    reversible=True,
    side_effecting=False,
    requires_authority_rank=5,
    required_permission="read_analytics",
)
SERVER = McpServerSettings(name="analytics", namespace="analytics", envelope=READ_ONLY)


@pytest.fixture
def parquet(tmp_path: Path) -> Path:
    store = DuckDbObservabilityStore(tmp_path / "observability.duckdb")
    DuckDbFactLedger(store).record(LlmCallFact(model="m"))
    directory = tmp_path / "parquet"
    ParquetExporter(store, parquet_dir=directory, evals_db=tmp_path / "none.duckdb").export()
    return directory


def _source(parquet: Path, server: McpServerSettings = SERVER) -> GovernedMcpToolSource:
    transport = SdkMcpTransport(lambda spec: build_server(SourceDatabase(parquet)))
    return GovernedMcpToolSource(
        server_spec(server), transport, ConfiguredGovernancePolicy([server])
    )


async def test_the_servers_tools_arrive_namespaced_under_the_local_envelope(parquet: Path) -> None:
    source = _source(parquet)
    try:
        tools = {tool.qualified_name: tool for tool in await source.tools()}
    finally:
        await source.close()
    assert set(tools) == {"analytics.list_views", "analytics.describe_view", "analytics.run_query"}
    query = tools["analytics.run_query"]
    assert (query.side_effecting, query.surface, query.required_permission) == (
        False,
        "locked",
        "read_analytics",
    )


async def test_an_undeclared_server_gets_the_most_restrictive_envelope(parquet: Path) -> None:
    undeclared = McpServerSettings(name="other", namespace="other")
    source = _source(parquet, undeclared)
    try:
        tool = (await source.tools())[0]
    finally:
        await source.close()
    default = get_settings().mcp.default_envelope
    assert (tool.side_effecting, tool.reversible, tool.surface) == (
        default.side_effecting,
        default.reversible,
        default.surface,
    )


async def test_a_call_goes_to_the_server_and_comes_back_as_a_tool_result(parquet: Path) -> None:
    source = _source(parquet)
    try:
        registry = NamespacedToolRegistry()
        registry.register_all(await source.tools())
        tool = registry.get("analytics.run_query")
        assert tool is not None
        result = await tool.handler(sql="SELECT count(*) AS n FROM llm_calls")
        refused = await tool.handler(sql="DROP TABLE llm_calls")
    finally:
        await source.close()
    assert result.success and result.data["rows"] == [[1]]
    assert not refused.success and refused.error


async def test_the_registry_validates_arguments_against_the_tools_schema(parquet: Path) -> None:
    source = _source(parquet)
    try:
        registry = NamespacedToolRegistry()
        registry.register_all(await source.tools())
    finally:
        await source.close()
    assert registry.build_call("analytics.run_query", {"sql": "SELECT 1"}) is not None
    assert registry.build_call("analytics.run_query", {}) is None


async def test_a_closed_session_refuses_calls(parquet: Path) -> None:
    transport = SdkMcpTransport(lambda spec: build_server(SourceDatabase(parquet)))
    session = await transport.connect(McpServerSpec(name="analytics", namespace="analytics"))
    await session.close()
    with pytest.raises(McpSessionClosedError):
        await session.list_tools()


async def test_an_unreachable_server_fails_to_connect() -> None:
    def unreachable(spec: McpServerSpec) -> str:
        return "http://127.0.0.1:9/mcp"

    with pytest.raises(McpSessionClosedError):
        await SdkMcpTransport(unreachable).connect(McpServerSpec(name="x", namespace="x"))


def test_an_argument_model_follows_the_schema() -> None:
    model = arguments_model(
        "t",
        {
            "properties": {"sql": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["sql"],
        },
    )
    assert model(sql="SELECT 1").model_dump() == {"sql": "SELECT 1", "limit": None}
    with pytest.raises(ValidationError):
        model(limit=3)
    with pytest.raises(ValidationError):
        model(sql="x", unexpected=True)
