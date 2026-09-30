"""The composition root (phase-2 scaffold).

The agent answers every message with its greeting, so the shell, the
overlay and the Makefile registry can be exercised end to end before any
research logic exists.
"""

from dataclasses import dataclass

from agentic_commons.config import FeatureSet, active_features, get_settings
from agentic_commons.observability import StageReporter
from research_agent.config import get_agent_settings


@dataclass(frozen=True)
class TurnResult:
    """What one turn returned to the shell."""

    response: str


class ScaffoldResearchAgent:
    """Answers with the greeting; replaced by the real graphs."""

    def __init__(self, reporter: StageReporter | None, features: FeatureSet) -> None:
        self._reporter = reporter
        self.features = features

    async def handle(self, query: str, session_id: str) -> TurnResult:
        if self._reporter is not None:
            self._reporter.completed("intake", fields={"session": session_id})
        return TurnResult(response=get_agent_settings().greeting)


def build_agent(
    reporter: StageReporter | None = None, features: FeatureSet | None = None
) -> ScaffoldResearchAgent:
    """Wire the research agent from config, honouring anything injected."""
    posture = features if features is not None else active_features(get_settings().features)
    return ScaffoldResearchAgent(reporter, posture)
