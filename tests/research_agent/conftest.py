"""Fixtures for the research agent's tests.

The shared conftest runs every test under the customer support overlay
(importing that package declares it). The research agent's tests layer its
own overlay on top for their duration and restore the configuration
afterwards, so no other test ever sees research settings.

No test touches the network or a real model: the web is the recorded
fixture world replayed from a temporary directory, and every role is the
fixture world's reference model (`model_backend: fixture`).
"""

import asyncio
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from agentic_commons.artifacts import DirectoryArtifactStore
from agentic_commons.audit import SqliteAuditLedger
from agentic_commons.config import (
    REPO_ROOT,
    FeatureSet,
    clear_overlays,
    get_settings,
    reload_settings,
    set_active_features,
    use_overlay,
)
from agentic_commons.durability import create_checkpointer, model_types
from agentic_commons.memory import EpisodicStore
from agentic_commons.observability import InMemoryLiveStatusFeed, StageReporter
from agentic_commons.persistence import IN_MEMORY, SqliteConnectionPool
from agentic_commons.web import FixtureStore, WebGateway
from research_agent.config import get_agent_settings, reload_agent_settings
from research_agent.fixture_world.generate import record
from research_agent.fixture_world.scripted_user import ScriptedUser, UserScript
from research_agent.fixture_world.world import load_world
from research_agent.graph import build_agent
from research_agent.graph.agent import ResearchAgent, TurnResult
from research_agent.graph.state import StateTypes

RA_OVERLAY = REPO_ROOT / "src" / "research_agent" / "config" / "config.yaml"
_RECORDED: dict[str, FixtureStore] = {}

DECISION_ANSWERS = {
    "decision": "Pick one chemistry for the 2027 microgrid tender by March",
    "options": "Veltrane VX40, Orrin FerroCell, Calder C-280",
    "criteria": "cost, cycle life, cold-weather performance, safety",
    "constraints": "none",
    "purpose": "Feeds the village council procurement decision",
    "recency": "any",
    "sources": "primary sources preferred",
    "context": "Tamsin Co-op; budget 2 million",
    "deliverable": "a short memo",
}
DECISION_REQUEST = "Help me choose a grid storage option for the Tamsin Valley microgrid"


@pytest.fixture
def ra_config() -> Iterator[None]:
    """The research agent's overlay, applied for one test."""
    use_overlay(RA_OVERLAY)
    reload_settings()
    reload_agent_settings()
    set_active_features(None)
    yield
    clear_overlays()
    set_active_features(None)
    reload_settings()


@pytest.fixture
def fixture_web(ra_config: None, tmp_path_factory: pytest.TempPathFactory) -> FixtureStore:
    """The fixture world, recorded once per session through the real adapters."""
    if "store" not in _RECORDED:
        world = load_world(get_agent_settings().resolved_fixture_world_path)
        root = tmp_path_factory.mktemp("web_fixtures")
        max_results = get_agent_settings().research.max_results_per_search
        _RECORDED["store"] = asyncio.run(record(world, root, max_results))
    return _RECORDED["store"]


@dataclass
class Harness:
    """One built agent and the stores a test inspects."""

    agent: ResearchAgent
    feed: InMemoryLiveStatusFeed
    audit: SqliteAuditLedger
    artifacts: DirectoryArtifactStore
    gateways: list[WebGateway]

    def stages(self) -> list[str]:
        return [event.stage for event in self.feed.events]

    def audited(self, action: str) -> list[object]:
        return [
            entry
            for entry in self.audit.by_decision("allow")
            + self.audit.by_decision("deny")
            + self.audit.by_decision("escalate")
            if entry.action == action
        ]


BuildHarness = Callable[..., Harness]


@pytest.fixture
def harness(ra_config: None, fixture_web: FixtureStore, tmp_path: Path) -> BuildHarness:
    """Build the research agent with a replayed web, in-memory and tmp stores."""

    def build(features: dict[str, bool] | None = None, gateway_mode: str = "replay") -> Harness:
        posture: FeatureSet = get_settings().features
        for path, value in (features or {}).items():
            posture = posture.with_override(path, value)
        pool = SqliteConnectionPool(IN_MEMORY)
        audit = SqliteAuditLedger(pool, agent_id=get_settings().agent.id)
        feed = InMemoryLiveStatusFeed(max_events=0)
        gateways: list[WebGateway] = []

        def gateway() -> WebGateway:
            gateways.append(WebGateway("replay", fixtures=fixture_web))
            return gateways[-1]

        artifacts = DirectoryArtifactStore(tmp_path / "artifacts")
        agent = build_agent(
            reporter=StageReporter(feed, "test"),
            features=posture,
            gateway=gateway,
            pool=pool,
            checkpointer=create_checkpointer(
                state_types=model_types(StateTypes), path=tmp_path / "ckpt.db"
            ),
            audit=audit,
            artifacts=artifacts,
            episodic=EpisodicStore(tmp_path / "episodic"),
        )
        return Harness(agent=agent, feed=feed, audit=audit, artifacts=artifacts, gateways=gateways)

    return build


async def converse(
    h: Harness, script: UserScript, session_id: str = "s1"
) -> tuple[TurnResult, ScriptedUser]:
    """Play a script to delivery; return the final turn and the user (with transcript)."""
    agent_settings = get_agent_settings()
    user = ScriptedUser(script, agent_settings.command_names, agent_settings.scoping.max_turns * 2)
    return await user.converse(h.agent, session_id), user


def decision_script(**overrides: object) -> UserScript:
    data: dict[str, object] = {"first_message": DECISION_REQUEST, "answers": dict(DECISION_ANSWERS)}
    data.update(overrides)
    return UserScript.model_validate(data)
