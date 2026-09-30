"""The research agent, presented to the operator shell (phase-2 scaffold).

Everything the shell must say in this agent's vocabulary is gathered
here, behind the platform's `AgentHost` protocol. The platform composes
it from `agent.host` in this agent's overlay and never imports it.
"""

from collections.abc import Mapping
from typing import Any

from agentic_commons.commands.types import Command
from agentic_commons.config import FeatureSet
from agentic_commons.observability import StageReporter
from research_agent.config import get_agent_settings


class ResearchHost:
    """The research agent, as the shell sees it."""

    def title(self) -> str:
        return get_agent_settings().title

    def placeholder(self) -> str:
        return get_agent_settings().placeholder

    def prepare(self) -> None:
        """Nothing slow to warm yet."""

    def build(self, reporter: StageReporter, features: FeatureSet | None = None) -> Any:
        from research_agent.graph import build_agent

        return build_agent(reporter=reporter, features=features)

    def commands(self) -> list[Command]:
        return []

    def stage_labels(self) -> Mapping[str, str]:
        return {}


__all__ = ["ResearchHost"]
