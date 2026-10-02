"""Fixtures for the research agent's tests.

The shared conftest runs every test under the customer support overlay
(importing that package declares it). The research agent's tests layer its
own overlay, then `offline.yaml`, on top for their duration and restore the
configuration afterwards, so no other test ever sees research settings.

No test touches the network or a real model: every role is a scripted model
injected through `build_agent(clients=...)`, the web is an empty replay store,
and the classifiers and hosted guards are withheld from the posture.
"""

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from agentic_commons.llm.client import LLMClient
from agentic_commons.memory import EpisodicStore
from agentic_commons.observability import InMemoryLiveStatusFeed, StageReporter
from agentic_commons.persistence import IN_MEMORY, SqliteConnectionPool
from agentic_commons.web import FixtureStore, WebGateway
from research_agent.config import reload_agent_settings
from research_agent.graph import build_agent
from research_agent.graph.agent import ResearchAgent
from research_agent.graph.state import StateTypes
from tests.conftest import ScriptedLLM

RA_OVERLAY = REPO_ROOT / "src" / "research_agent" / "config" / "config.yaml"
OFFLINE = Path(__file__).with_name("offline.yaml")
# Withheld in every harness unless a test turns them on: each would load a local model or
# call a hosted one.
OFFLINE_POSTURE = {
    "gate_encoder": False,
    "pii": False,
    "prompt_injection": False,
    "content_moderation": False,
    "injection_classifier.model": False,
    "direct_answer": False,
}


@pytest.fixture
def ra_config() -> Iterator[None]:
    """The research agent's overlay plus the offline test overlay, for one test."""
    use_overlay(RA_OVERLAY)
    use_overlay(OFFLINE)
    reload_settings()
    reload_agent_settings()
    set_active_features(None)
    yield
    clear_overlays()
    set_active_features(None)
    reload_settings()


class ScriptedRoles:
    """`RoleClients` over scripted models: each role answers from its own script."""

    def __init__(self, scripts: Mapping[str, Sequence[Any]]) -> None:
        self.models = {
            role: ScriptedLLM([_step(s) for s in steps]) for role, steps in scripts.items()
        }
        self._silent = ScriptedLLM()

    def client_for(self, role: str) -> LLMClient:
        return self.models.get(role, self._silent)

    def model_key(self, role: str) -> str:
        return f"scripted:{role}"


def _step(step: Any) -> dict[str, Any]:
    """A script step: a model answer as a mapping (sent as JSON) or a raw step dict."""
    if isinstance(step, dict) and ("content" in step or "tool_calls" in step):
        return step
    return {"content": json.dumps(step)}


@dataclass
class Harness:
    """One built agent and the stores a test inspects."""

    agent: ResearchAgent
    feed: InMemoryLiveStatusFeed
    audit: SqliteAuditLedger
    roles: ScriptedRoles

    def stages(self) -> list[str]:
        return [event.stage for event in self.feed.events]


BuildHarness = Callable[..., Harness]


@pytest.fixture
def harness(ra_config: None, tmp_path: Path) -> BuildHarness:
    """Build the research agent on scripted roles, an empty replay web and tmp stores."""

    def build(
        scripts: Mapping[str, Sequence[Any]] | None = None,
        features: dict[str, bool] | None = None,
    ) -> Harness:
        posture: FeatureSet = get_settings().features
        for path, value in {**OFFLINE_POSTURE, **(features or {})}.items():
            posture = posture.with_override(path, value)
        pool = SqliteConnectionPool(IN_MEMORY)
        audit = SqliteAuditLedger(pool, agent_id=get_settings().agent.id)
        feed = InMemoryLiveStatusFeed(max_events=0)
        roles = ScriptedRoles(scripts or {})
        agent = build_agent(
            reporter=StageReporter(feed, "test"),
            features=posture,
            clients=roles,
            gateway=lambda: WebGateway("replay", fixtures=FixtureStore(tmp_path / "web", "none")),
            pool=pool,
            checkpointer=create_checkpointer(
                state_types=model_types(StateTypes), path=tmp_path / "ckpt.db"
            ),
            audit=audit,
            artifacts=DirectoryArtifactStore(tmp_path / "artifacts"),
            episodic=EpisodicStore(tmp_path / "episodic"),
        )
        return Harness(agent=agent, feed=feed, audit=audit, roles=roles)

    return build
