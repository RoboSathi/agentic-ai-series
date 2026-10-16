"""This agent's eval plugin — everything agent-specific, behind one seam.

Named in this agent's config (`evals.plugin`), which is how the platform's
eval runner finds it without importing it. It hands the platform:

- the golden cases and the isolated agent that answers them;
- the oracle (this agent's business rules on the platform's `InvariantOracle`);
- the DeepEval metrics, and the frame (role, scenario, expected outcome) each
  case is judged against;
- the Promptfoo red-team config;
- the intent classifier bake-off.
"""

from pathlib import Path
from typing import Any

import yaml

from agent_platform.config import get_settings, resolve_path
from agent_platform.core.primitives.hashing import canonical_hash
from agent_platform.evals.core.contracts import (
    AgentUnderTest,
    ConversationFrame,
    GoldenCase,
    Grader,
    QualityMetricSpec,
)
from agent_platform.evals.intent import BakeoffResult
from customer_support_agent.config.settings import CsaEvalSettings, get_agent_settings
from customer_support_agent.evals.agent_under_test import CsaAgentUnderTest
from customer_support_agent.evals.dataset.schema import CsaCasePayload, load_golden_cases
from customer_support_agent.evals.fixture import EvalFixture
from customer_support_agent.evals.oracle import csa_oracle

_METRICS_KEY = "metrics"


class UnknownSuiteError(ValueError):
    """A suite this agent does not declare cases for."""


class CsaEvalPlugin:
    """The Customer Support agent's evals."""

    def __init__(self, settings: CsaEvalSettings | None = None) -> None:
        self._app_config = get_settings()
        self._settings = settings or get_agent_settings().evals
        self._fixture = EvalFixture(self._settings)
        self._classifier: object | None = None
        self._runs = 0

    @property
    def agent_id(self) -> str:
        return self._app_config.agent.id

    def load_cases(self, suite: str) -> list[GoldenCase]:
        if suite != self._app_config.evals.suites.golden:
            raise UnknownSuiteError(
                f"'{suite}' has no golden cases here; the red-team attacks live in "
                f"{self._settings.security_config_path}"
            )
        return load_golden_cases(self._settings.resolved_dataset_path)

    def prepare(self) -> None:
        """Build the fixture and warm native models, before any event loop."""
        from agent_platform.scoring.entailment import preload_entailment_model
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
        return csa_oracle(self._fixture.facts(), self._settings)

    def quality_metrics(self) -> list[QualityMetricSpec]:
        data = _yaml(self._settings.quality_metrics_path)
        return [QualityMetricSpec.model_validate(item) for item in data.get(_METRICS_KEY, [])]

    def conversation_frame(self, case: GoldenCase) -> ConversationFrame:
        payload = CsaCasePayload.of(case)
        return ConversationFrame(
            chatbot_role=self._settings.chatbot_role,
            scenario=payload.scenario,
            expected_outcome=payload.expected_outcome,
        )

    def security_config(self) -> str:
        return self._settings.security_config_path

    def fingerprint_inputs(self) -> dict[str, str]:
        """The agent's model, and the fixture world every case runs in."""
        llm = self._app_config.llm
        seed = self._settings.resolved_fixture_seed_path.read_text(encoding="utf-8")
        return {"model": llm.provider_settings().model, "fixture": canonical_hash(seed)}

    async def intent_bakeoff(self) -> BakeoffResult:
        from customer_support_agent.evals.intent.bakeoff import run

        return await run()


def _yaml(path: str) -> dict[str, Any]:
    loaded = yaml.safe_load(Path(resolve_path(path)).read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}
