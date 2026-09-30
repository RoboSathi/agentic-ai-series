"""The research agent's eval plugin — everything agent-specific, behind one seam.

Named in this agent's overlay (`evals.plugin`), which is how the platform's
eval runner finds it without importing it. Two optional hooks go with it:
its security suite is native (golden-style cases over the fixture world's
hostile pages, graded by this oracle, offline) rather than Promptfoo, and
under the fixture backend its judge stage uses the offline reference judge.
"""

from typing import Any

import yaml

from agentic_commons.config import get_settings
from agentic_commons.evals.contracts import (
    AgentUnderTest,
    GoldenCase,
    Grader,
    QualityMetricSpec,
    Rubric,
)
from agentic_commons.web import FixtureStore
from research_agent.config import get_agent_settings
from research_agent.config.settings import ResearchAgentSettings
from research_agent.evals.agent_under_test import ResearchAgentUnderTest
from research_agent.evals.dataset.schema import load_cases
from research_agent.evals.oracle import ResearchOracle
from research_agent.evals.reference_judge import ReferenceJudge
from research_agent.fixture_world.world import load_world


class UnknownSuiteError(ValueError):
    """A suite this agent does not declare cases for."""


class LiveEvalError(RuntimeError):
    """Evals were started against the live web or real models (ADR ra-0009)."""


class ResearchEvalPlugin:
    """The Research Agent's evals."""

    def __init__(self, settings: ResearchAgentSettings | None = None) -> None:
        self._settings = settings or get_agent_settings()
        self._fixtures: FixtureStore | None = None
        self._runs = 0

    @property
    def agent_id(self) -> str:
        return get_settings().agent.id

    def load_cases(self, suite: str) -> list[GoldenCase]:
        files = self._settings.evals.suite_files
        if suite not in files:
            raise UnknownSuiteError(f"'{suite}' is not one of this agent's suites: {sorted(files)}")
        return load_cases(self._settings.evals.resolved_dataset_dir / files[suite], suite)

    def prepare(self) -> None:
        """Refuse a live posture, then record the fixture world if it never was."""
        from research_agent.fixture_world.generate import ensure_recorded

        if get_settings().web.mode != "replay" or self._settings.model_backend != "fixture":
            raise LiveEvalError(
                "research evals are fixtures-only: run them with `make eval ra` "
                "(which applies config/profiles/demo.yaml)"
            )
        self._fixtures = ensure_recorded()

    def agent_under_test(self) -> AgentUnderTest:
        self._runs += 1
        return ResearchAgentUnderTest(self._store(), self._settings, run_key=f"run-{self._runs}")

    def oracle(self) -> Grader:
        world = load_world(self._settings.resolved_fixture_world_path)
        hostile = frozenset(page.url for page in world.hostile_pages())
        return ResearchOracle(self._store(), hostile, self._settings.evals)

    def judge_rubric(self, suite: str) -> Rubric:
        path = self._settings.evals.resolved_judge_rubric_path
        return Rubric.model_validate(_yaml(path.read_text()))

    def quality_metrics(self) -> list[QualityMetricSpec]:
        """None: DeepEval's metrics need a hosted model, and these evals are offline."""
        return []

    def security_config(self) -> str:
        return self._settings.evals.security_suite

    def fingerprint_inputs(self) -> dict[str, str]:
        roles = get_settings().llm.roles
        return {
            "model": self._settings.model_backend,
            "roles": ",".join(f"{r}={s.provider}:{s.model}" for r, s in sorted(roles.items())),
            "fixture_set": get_settings().web.fixture_set,
        }

    # -- optional hooks ------------------------------------------------------------------

    def native_security_suite(self) -> str:
        return self._settings.evals.security_suite

    def judge_client(self) -> Any:
        if self._settings.model_backend != "fixture":
            return None
        return ReferenceJudge(
            self.judge_rubric(self._settings.evals.security_suite),
            self._settings.evals.reference_judge,
            self._settings.synthesis.citation_pattern,
        )

    def _store(self) -> FixtureStore:
        if self._fixtures is None:
            web = get_settings().web
            self._fixtures = FixtureStore(web.resolved_fixtures_dir, web.fixture_set)
        return self._fixtures


def _yaml(text: str) -> dict[str, Any]:
    loaded = yaml.safe_load(text)
    return loaded if isinstance(loaded, dict) else {}
