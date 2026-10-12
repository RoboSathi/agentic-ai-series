"""Every MCP protocol event is reported; the ones an operator acts on become cards.

The server is the platform's real analytics server over a real exported Parquet,
connected in-process; the feed is a real in-memory one.
"""

from pathlib import Path
from typing import Any

import pytest

from agent_platform.analytics import ParquetExporter
from agent_platform.analytics.source_db import SourceDatabase
from agent_platform.config.settings import McpServerSettings, McpToolEnvelope
from agent_platform.knowledge.mcp import (
    ConfiguredGovernancePolicy,
    GovernedMcpToolSource,
    SdkMcpTransport,
    server_spec,
)
from agent_platform.knowledge.mcp.client.events import (
    MCP_CALL,
    MCP_NOTIFICATION,
    MCP_SERVER,
    MCP_SESSION,
    TOOLS_LIST_CHANGED,
    McpEvents,
)
from agent_platform.knowledge.mcp.common.types import McpServerInfo
from agent_platform.knowledge.mcp.server.analytics import build_server
from agent_platform.observability import InMemoryLiveStatusFeed, StageReporter
from agent_platform.observability.facts import DuckDbFactStore, ModelCallFact
from agent_platform.observability.store import DuckDbObservabilityStore

READ_ONLY = McpToolEnvelope(
    surface="locked", reversible=True, side_effecting=False, requires_authority_rank=5
)
SERVER = McpServerSettings(name="analytics", namespace="analytics", envelope=READ_ONLY)


class Recorded(McpEvents):
    """The real sink, remembering every server log it was told about."""

    def __init__(self, reporter: StageReporter) -> None:
        super().__init__(reporter)
        self.logs: list[tuple[str, str]] = []

    def server_log(self, server: str, level: str, logger: str, data: Any) -> None:
        self.logs.append((level, str(data)))
        super().server_log(server, level, logger, data)


@pytest.fixture
def parquet(tmp_path: Path) -> Path:
    store = DuckDbObservabilityStore(tmp_path / "observability.duckdb")
    DuckDbFactStore(store).append(ModelCallFact(model="m"))
    directory = tmp_path / "parquet"
    ParquetExporter(store, parquet_dir=directory, evals_db=tmp_path / "none.duckdb").export()
    return directory


@pytest.fixture
def feed() -> InMemoryLiveStatusFeed:
    return InMemoryLiveStatusFeed()


def _source(parquet: Path, events: McpEvents) -> GovernedMcpToolSource:
    transport = SdkMcpTransport(lambda spec: build_server(SourceDatabase(parquet)))
    return GovernedMcpToolSource(
        server_spec(SERVER), transport, ConfiguredGovernancePolicy([SERVER]), events
    )


def _cards(feed: InMemoryLiveStatusFeed, stage: str) -> list[Any]:
    return [event for event in feed.events if event.stage == stage]


async def test_the_handshake_is_one_server_card_with_how_it_was_negotiated(
    parquet: Path, feed: InMemoryLiveStatusFeed
) -> None:
    source = _source(parquet, McpEvents(StageReporter(feed, run_id="t")))
    try:
        await source.tools()
    finally:
        await source.close()
    [card] = _cards(feed, MCP_SERVER)
    assert card.status == "completed"
    assert "5 tools offered" in (card.summary or "")
    handshake = card.fields["Handshake"]
    assert handshake[0].split(" · ")[0] in {"initialize", "server/discover"}
    assert handshake[1] == "client: agentic-harness 1.0"


async def test_a_refused_query_reaches_the_client_as_a_warning_notification(
    parquet: Path, feed: InMemoryLiveStatusFeed
) -> None:
    events = Recorded(StageReporter(feed, run_id="t"))
    source = _source(parquet, events)
    try:
        tools = {tool.qualified_name: tool for tool in await source.tools()}
        result = await tools["analytics.run_query"].handler(sql="SELECT agnet_id FROM model_calls")
        empty = await tools["analytics.run_query"].handler(
            sql="SELECT * FROM model_calls WHERE 1=0"
        )
    finally:
        await source.close()
    assert not result.success and "Did you mean: agent_id?" in (result.error or "")
    assert [level for level, _ in events.logs] == ["warning", "info"]
    # Only the warning is a card; the empty result's note is in the log and the result.
    [card] = _cards(feed, MCP_NOTIFICATION)
    assert card.status == "warning" and "\n" not in (card.summary or "")
    assert empty.data["notes"] and empty.data["notes"][0].startswith("0 rows")


async def test_a_call_after_the_session_closed_is_a_failed_mcp_call_card(
    parquet: Path, feed: InMemoryLiveStatusFeed
) -> None:
    source = _source(parquet, McpEvents(StageReporter(feed, run_id="t")))
    tools = {tool.qualified_name: tool for tool in await source.tools()}
    session = source._session
    assert session is not None
    await session.close()
    with pytest.raises(Exception):
        await tools["analytics.list_views"].handler()
    [card] = _cards(feed, MCP_CALL)
    assert card.status == "failed" and "no answer" in (card.summary or "")


async def test_a_server_that_cannot_be_reached_is_a_failed_server_card(
    feed: InMemoryLiveStatusFeed,
) -> None:
    transport = SdkMcpTransport(lambda spec: "http://127.0.0.1:9/mcp")
    events = McpEvents(StageReporter(feed, run_id="t"))
    source = GovernedMcpToolSource(
        server_spec(SERVER), transport, ConfiguredGovernancePolicy([SERVER]), events
    )
    with pytest.raises(Exception):
        await source.tools()
    statuses = [card.status for card in _cards(feed, MCP_SERVER)]
    assert statuses == ["failed"]
    assert [card.status for card in _cards(feed, MCP_SESSION)] == ["failed"]


def test_only_what_an_operator_acts_on_becomes_a_card(feed: InMemoryLiveStatusFeed) -> None:
    events = McpEvents(StageReporter(feed, run_id="t"))
    events.opening("analytics", "target")
    events.connected("analytics", McpServerInfo(name="analytics"))
    events.progress("analytics", "run_query", 1, 2, "half")
    events.server_log("analytics", "info", "srv", "fine")
    events.notification("analytics", "notifications/resources/updated", {})
    events.closed("analytics")
    assert feed.events == []
    events.notification("analytics", TOOLS_LIST_CHANGED, {})
    events.server_log("analytics", "error", "srv", "broken\nsecond line")
    shown = [(card.stage, card.status) for card in feed.events]
    assert shown == [(MCP_NOTIFICATION, "warning"), (MCP_NOTIFICATION, "failed")]
    assert feed.events[1].fields["Message"] == ["broken", "second line"]


def test_without_a_feed_every_event_is_still_logged() -> None:
    events = McpEvents(None)
    events.connect_failed("analytics", "refused")
    events.session_failed("analytics", "gone")
