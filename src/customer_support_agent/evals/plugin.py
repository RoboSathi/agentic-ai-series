"""This agent's eval plugin — everything agent-specific, behind one seam.

Named in this agent's config (`evals.plugin`), which is how the platform's
eval runner finds it without importing it.
"""

from pathlib import Path
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
from customer_support_agent.config.settings import CsaEvalSettings, get_agent_settings
from customer_support_agent.evals.agent_under_test import CsaAgentUnderTest
from customer_support_agent.evals.dataset.schema import load_golden_cases
from customer_support_agent.evals.fixture import EvalFixture
from customer_support_agent.evals.oracle import CsaOracle

_GOLDEN_SUITE = "golden"


class UnknownSuiteError(ValueError):
    """A suite this agent does not declare cases for."""


class CsaEvalPlugin:
    """The Customer Support agent's evals."""

    def __init__(self, settings: CsaEvalSettings | None = None) -> None:
        self._settings = settings or get_agent_settings().evals
        self._fixture = EvalFixture(self._settings)
        self._classifier: object | None = None
        self._runs = 0

    @property
    def agent_id(self) -> str:
        return get_settings().agent.id

    def load_cases(self, suite: str) -> list[GoldenCase]:
        if suite != _GOLDEN_SUITE:
            raise UnknownSuiteError(
                f"'{suite}' has no golden cases here; security cases live in "
                f"{self._settings.security_config_path}"
            )
        return load_golden_cases(self._settings.resolved_dataset_path)

    def prepare(self) -> None:
        """Build the fixture and warm native models, before any event loop."""
        from agentic_commons.evals.entailment import preload_entailment_model
        from customer_support_agent.intent.router import load_intent_classifier

        self._fixture.ensure_built()
        self._classifier = load_intent_classifier()
        preload_entailment_model()

    def agent_under_test(self) -> AgentUnderTest:
        self._runs += 1
        return CsaAgentUnderTest(
            self._fixture, run_key=f"run-{self._runs}", classifier=self._classifier
        )

    def oracle(self) -> Grader:
        return CsaOracle(self._fixture.facts())

    def judge_rubric(self, suite: str) -> Rubric:
        return Rubric.model_validate(_yaml(Path(self._settings.judge_rubric_path)))

    def quality_metrics(self) -> list[QualityMetricSpec]:
        data = _yaml(Path(self._settings.quality_metrics_path))
        return [QualityMetricSpec.model_validate(item) for item in data.get("metrics", [])]

    def security_config(self) -> str:
        return self._settings.security_config_path

    def fingerprint_inputs(self) -> dict[str, str]:
        llm = get_settings().llm
        return {"model": llm.provider_settings().model}


def _yaml(path: Path) -> dict[str, Any]:
    from agentic_commons.config import resolve_path

    loaded = yaml.safe_load(resolve_path(str(path)).read_text())
    return loaded if isinstance(loaded, dict) else {}
