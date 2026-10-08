"""Fixtures for the ops agent's tests.

The shared conftest runs every test under the customer support overlay; these
tests layer the ops agent's overlay on top for their duration. The model is
scripted, the MCP server is the platform's real analytics server over a real
exported mart, connected in-process; every ledger is a real DuckDB in tmp.
"""

import json
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from agent_platform.analytics import MartExporter
from agent_platform.analytics.mart_db import MartDatabase
from agent_platform.analytics.mcp_server import build_server
from agent_platform.audit import AuditEntry, DuckDbAuditLedger
from agent_platform.config import (
    REPO_ROOT,
    FeatureSet,
    clear_overlays,
    get_settings,
    reload_settings,
    set_active_features,
    use_overlay,
)
from agent_platform.llm.types import ModelToolCall
from agent_platform.mcp import McpServerSpec, SdkMcpTransport
from agent_platform.observability import InMemoryLiveStatusFeed, StageReporter
from agent_platform.observability.store import DuckDbObservabilityStore
from agent_platform.persistence import IN_MEMORY, SqliteConnectionPool
from ops_agent.config import reload_agent_settings
from ops_agent.graph.agent import OpsAgent
from ops_agent.graph.builder import build_agent
from tests.conftest import ScriptedLLM

OPS_OVERLAY = REPO_ROOT / "src" / "ops_agent" / "config" / "config.yaml"
COUNT_SQL = "SELECT decision, count(*) AS n FROM audit_decisions GROUP BY 1 ORDER BY 1"


@pytest.fixture
def ops_config() -> Iterator[None]:
    use_overlay(OPS_OVERLAY)
    reload_settings()
    reload_agent_settings()
    set_active_features(None)
    yield
    clear_overlays()
    set_active_features(None)
    reload_settings()


@pytest.fixture
def mart(tmp_path: Path, ops_config: None) -> Path:
    """A mart holding three audit decisions: two allows and a deny."""
    store = DuckDbObservabilityStore(tmp_path / "source.duckdb")
    ledger = DuckDbAuditLedger(store, agent_id="some_agent")
    for decision in ("allow", "allow", "deny"):
        ledger.record(AuditEntry(trace_id="t", actor="a", action="x.y", decision=decision))
    directory = tmp_path / "mart"
    MartExporter(store, mart_dir=directory, evals_db=tmp_path / "none.duckdb").export()
    return directory


def query_call(sql: str = COUNT_SQL, call_id: str = "c1") -> list[ModelToolCall]:
    return [
        ModelToolCall(id=call_id, name="analytics__run_query", arguments=json.dumps({"sql": sql}))
    ]


@dataclass
class Harness:
    agent: OpsAgent
    llm: ScriptedLLM
    store: DuckDbObservabilityStore
    feed: InMemoryLiveStatusFeed

    def rows(self, sql: str) -> list[tuple[Any, ...]]:
        return [tuple(row.values()) for row in self.store.iter_rows(sql)]


@pytest.fixture
async def harness(mart: Path, tmp_path: Path) -> AsyncIterator[Any]:
    agents: list[OpsAgent] = []

    def build(
        script: list[dict[str, Any]], features: dict[str, bool] | None = None, down: bool = False
    ) -> Harness:
        def target(spec: McpServerSpec) -> Any:
            return "http://127.0.0.1:9/mcp" if down else build_server(MartDatabase(mart))

        store = DuckDbObservabilityStore(tmp_path / f"ops-{len(agents)}.duckdb")
        feed = InMemoryLiveStatusFeed()
        posture = FeatureSet(tree=get_settings().features.tree)
        for path, on in (features or {}).items():
            posture = posture.with_override(path, on)
        llm = ScriptedLLM(script)
        agent = build_agent(
            reporter=StageReporter(feed, run_id="ops-test"),
            features=posture,
            llm=llm,
            transport=SdkMcpTransport(target),
            store=store,
            pool=SqliteConnectionPool(IN_MEMORY),
        )
        agents.append(agent)
        return Harness(agent, llm, store, feed)

    yield build
    for agent in agents:
        await agent.close()
