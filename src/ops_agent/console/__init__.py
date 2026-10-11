"""The ops agent, presented to the operator shell (`agent.host` in its overlay)."""

from collections.abc import Mapping
from typing import Any

from agent_platform.commands.skills import SkillsCommand
from agent_platform.commands.types import Command
from agent_platform.config import FeatureSet
from agent_platform.memory.procedural.skills import SkillCatalog
from agent_platform.observability import StageReporter
from ops_agent.config import get_agent_settings


class OpsHost:
    """The ops agent, as the shell sees it."""

    def title(self) -> str:
        return get_agent_settings().title

    def placeholder(self) -> str:
        return get_agent_settings().placeholder

    def greeting(self) -> str:
        return get_agent_settings().greeting

    def prepare(self) -> None:
        """Nothing to warm: no local models, and the MCP server starts on the first turn."""

    def build(self, reporter: StageReporter, features: FeatureSet | None = None) -> Any:
        from ops_agent.graph.builder import build_agent

        self._agent = build_agent(reporter=reporter, features=features)
        return self._agent

    def commands(self) -> list[Command]:
        """`/skills` over this agent's skills; it has no curator queue to approve from."""
        return [SkillsCommand(self._catalog, lambda: None)]

    def _catalog(self) -> SkillCatalog | None:
        agent = getattr(self, "_agent", None)
        return agent.skill_catalog if agent is not None else None

    def stage_labels(self) -> Mapping[str, str]:
        return get_agent_settings().stage_labels


__all__ = ["OpsHost"]
