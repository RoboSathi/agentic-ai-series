"""The Eval Pyramid — four levels, four cadences (E09, LLD §18).

Component evals are cheap and run on every change; production shadow
evals are continuous and run against real traffic. The shape matters: a
suite that is all end-to-end tells you something broke without telling
you what, and a suite that is all unit tests tells you nothing broke
while the system fails.
"""

from typing import Awaitable, Callable, Protocol, runtime_checkable

from agent_platform.evals.manifest import EvalManifest, load_manifest
from agent_platform.evals.scoring import (
    NOLLM_TIERS,
    NollmScoringCascadeImpl,
    create_scoring_cascade,
)
from agent_platform.evals.types import (
    Claim,
    EvalCase,
    EvalOutcome,
    EvalReport,
    PyramidLevel,
)
from agent_platform.logging import get_logger

# A system under test: takes a case's input, returns what it produced.
SystemUnderTest = Callable[[EvalCase], Awaitable[str]]


@runtime_checkable
class EvalRunner(Protocol):
    """Runs a set of cases at one level of the pyramid."""

    level: PyramidLevel

    async def run(self, cases: list[EvalCase], system: SystemUnderTest) -> EvalReport: ...


class PyramidRunner:
    """Shared execution for every pyramid level (Template Method).

    The levels differ in *what* they exercise and how often, not in how a
    case is scored — so scoring lives here once, and each level is a thin
    subclass rather than a copy.
    """

    level: PyramidLevel = "component"

    def __init__(
        self,
        cascade: NollmScoringCascadeImpl | None = None,
        manifest: EvalManifest | None = None,
    ) -> None:
        self._cascade = cascade or create_scoring_cascade()
        self._manifest = manifest or load_manifest()
        self._logger = get_logger("evals")

    async def run(self, cases: list[EvalCase], system: SystemUnderTest) -> EvalReport:
        """Execute every case, score it through the cascade, and aggregate."""
        outcomes = [await self._run_case(case, system) for case in cases]
        return self._report(outcomes)

    async def _run_case(self, case: EvalCase, system: SystemUnderTest) -> EvalOutcome:
        """Run one case, turning a raised exception into a failed outcome."""
        try:
            produced = await system(case)
        except Exception as exc:
            self._logger.warning("Eval case raised", case_id=case.case_id, error=str(exc))
            return EvalOutcome(
                case_id=case.case_id,
                component=case.component,
                passed=False,
                score=0.0,
                tier="deterministic_rule",
                detail=f"system under test raised: {exc}",
            )
        verdict = await self._cascade.resolve(
            Claim(text=produced, evidence=case.evidence, expected=case.expected)
        )
        return EvalOutcome(
            case_id=case.case_id,
            component=case.component,
            passed=verdict.label == "supported",
            score=verdict.score,
            tier=verdict.tier,
            detail=verdict.rationale,
        )

    def _report(self, outcomes: list[EvalOutcome]) -> EvalReport:
        """Aggregate outcomes into a level report with its NOLLM ratio."""
        total = len(outcomes)
        pass_rate = sum(o.passed for o in outcomes) / total if total else 0.0
        nollm = sum(o.tier in NOLLM_TIERS for o in outcomes) / total if total else 1.0
        components = {o.component for o in outcomes}
        threshold = max(
            (self._manifest.threshold_for(component) for component in components), default=0.0
        )
        return EvalReport(
            level=self.level,
            outcomes=outcomes,
            pass_rate=pass_rate,
            nollm_scorer_ratio=nollm,
            regression_threshold=threshold,
        )


class ComponentEvalRunner(PyramidRunner):
    """Level 1 — one component in isolation. Runs on every change."""

    level: PyramidLevel = "component"


class IntegrationEvalRunner(PyramidRunner):
    """Level 2 — components wired together. Runs on every pull request."""

    level: PyramidLevel = "integration"


class GoldenTaskE2ERunner(PyramidRunner):
    """Level 3 — whole tasks against a curated golden set. Runs nightly."""

    level: PyramidLevel = "golden_task"


class ProductionShadowEvalRunner(PyramidRunner):
    """Level 4 — real traffic, scored but never served.

    Shadow evals are the only level whose input distribution is not
    curated, which is exactly why they catch what the other three cannot:
    the inputs nobody thought to write down.
    """

    level: PyramidLevel = "production_shadow"
