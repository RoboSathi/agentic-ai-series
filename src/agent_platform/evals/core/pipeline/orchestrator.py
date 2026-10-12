"""The eval pipeline: three graders over two suites, each ending in its report.

    golden    generate → oracle → DeepEval        → deepeval-<run_id>.md
    security  Promptfoo (attacks + assertions)    → promptfoo-<run_id>.md
    intent    the agent's classifier bake-off     → intent-<run_id>.md

Generation comes first (every grader reads its transcripts), then the oracle,
then DeepEval — always all three: a broken invariant is a finding to read
beside the quality scores, not a reason to withhold them. Each section is
timed, so the run ends with where the time went.
"""

import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ConfigDict, Field

from agent_platform.config import get_settings
from agent_platform.evals.core.contracts import (
    EvalPlugin,
    Grader,
    IntentBakeoffPlugin,
    QualityMetricSpec,
)
from agent_platform.evals.core.fingerprint import compute_fingerprint
from agent_platform.evals.core.pipeline.gate import FAIL, PASS, GateResult, evaluate_gate
from agent_platform.evals.core.pipeline.generate import GenerationStage
from agent_platform.evals.core.pipeline.grade import GradingStage
from agent_platform.evals.core.pipeline.runs import run_prefix
from agent_platform.evals.core.pipeline.trials import expand_trials
from agent_platform.evals.core.reporting import Overview
from agent_platform.evals.core.store import DuckDbEvalStore, RunRecord
from agent_platform.evals.core.store.records import new_run_id
from agent_platform.logging import get_logger

# The sections a run is timed by (named for display in `evals.reporting.overview.sections`).
AGENT_SECTION, ORACLE_SECTION, DEEPEVAL_SECTION = "agent", "oracle", "deepeval"
PROMPTFOO_SECTION, INTENT_SECTION = "promptfoo", "intent"


class NoRunError(LookupError):
    """A report was asked for, but no such run exists."""


class NoIntentBakeoffError(LookupError):
    """`STAGE=intent` for an agent that declares no bake-off."""


class StageOutcome(BaseModel):
    """What one suite produced: its gate, its report, and the overview to print."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: str
    gate: GateResult
    report: Path
    overview: Overview
    # Section → wall-clock seconds it took, in the order the sections ran.
    timings: dict[str, float] = Field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return self.gate.verdict == FAIL


class EvalPipeline:
    """Runs an agent's suites and writes their reports."""

    def __init__(self, plugin: EvalPlugin, store: DuckDbEvalStore | None = None) -> None:
        self._plugin = plugin
        self._store = store or DuckDbEvalStore()
        self._settings = get_settings().evals
        self._logger = get_logger("evals")

    @property
    def store(self) -> DuckDbEvalStore:
        return self._store

    async def golden(self) -> StageOutcome:
        """Generate, then grade with the oracle and with DeepEval — every section, every time."""
        timings: dict[str, float] = {}
        suite = self._settings.suites.golden
        cases = expand_trials(self._plugin.load_cases(suite))
        generation = GenerationStage(self._plugin, self._store)
        run = await _timed(timings, AGENT_SECTION, lambda: generation.run(suite, cases))
        by_id = self._store.cases_for_run(run.run_id)
        grading = GradingStage(self._store)
        oracle = [self._plugin.oracle()]
        await _timed(
            timings, ORACLE_SECTION, lambda: grading.run(run.run_id, "oracle", oracle, by_id)
        )
        metrics = self._plugin.quality_metrics()
        if metrics:
            judges = [self._deepeval(metrics)]
            await _timed(
                timings,
                DEEPEVAL_SECTION,
                lambda: grading.run(run.run_id, "deepeval", judges, by_id),
            )
        return self._golden_outcome(run, timings)

    def security(self) -> StageOutcome:
        """The agent's Promptfoo red-team suite (reused when nothing changed)."""
        from agent_platform.evals.promptfoo import PromptfooSuite
        from agent_platform.evals.promptfoo.report import PromptfooReport

        started = time.perf_counter()
        run = PromptfooSuite(self._plugin, self._store).run()
        timings = {PROMPTFOO_SECTION: time.perf_counter() - started}
        report = PromptfooReport(self._store)
        path = report.write(run)
        return StageOutcome(
            run_id=run.run_id,
            gate=evaluate_gate(self._store, run.run_id),
            report=path,
            overview=report.overview(run, path),
            timings=timings,
        )

    async def intent(self) -> StageOutcome:
        """The agent's intent classifier bake-off — offline, no provider calls."""
        from agent_platform.evals.intent import BakeoffResult
        from agent_platform.evals.intent import report as intent_report

        if not isinstance(self._plugin, IntentBakeoffPlugin):
            raise NoIntentBakeoffError(f"'{self._plugin.agent_id}' declares no intent bake-off.")
        started = time.perf_counter()
        result = BakeoffResult.model_validate(await self._plugin.intent_bakeoff())
        timings = {INTENT_SECTION: time.perf_counter() - started}
        fingerprint = compute_fingerprint(self._plugin.fingerprint_inputs())
        run_id = new_run_id(run_prefix(self._plugin), "intent", fingerprint.value)
        path = intent_report.write(result, run_id)
        gate = GateResult(verdict=PASS, soft_threshold=self._settings.gate.soft_pass_rate)
        return StageOutcome(
            run_id=run_id,
            gate=gate,
            report=path,
            overview=intent_report.overview(result, run_id, path),
            timings=timings,
        )

    def report(self, run_id: str | None = None) -> StageOutcome:
        """Rebuild a run's report from the store (default: the latest golden run)."""
        run = (
            self._store.get_run(run_id)
            if run_id
            else self._store.latest_run(self._settings.suites.golden)
        )
        if run is None:
            raise NoRunError(f"No eval run {run_id or 'yet'} — `make eval <agent>` records one.")
        if run.suite == self._settings.suites.security:
            from agent_platform.evals.promptfoo.report import PromptfooReport

            report = PromptfooReport(self._store)
            path = report.write(run)
            gate = evaluate_gate(self._store, run.run_id)
            return StageOutcome(
                run_id=run.run_id, gate=gate, report=path, overview=report.overview(run, path)
            )
        return self._golden_outcome(run, {})

    def _golden_outcome(self, run: RunRecord, timings: dict[str, float]) -> StageOutcome:
        from agent_platform.evals.deepeval.report import DeepEvalReport

        report = DeepEvalReport(self._store)
        path = report.write(run)
        return StageOutcome(
            run_id=run.run_id,
            gate=evaluate_gate(self._store, run.run_id),
            report=path,
            overview=report.overview(run, path),
            timings=timings,
        )

    def _deepeval(self, metrics: list[QualityMetricSpec]) -> Grader:
        # Imported here: DeepEval is the optional `evals` extra.
        from agent_platform.evals.deepeval import DeepEvalGrader

        return DeepEvalGrader(metrics, self._plugin.conversation_frame)


ResultT = TypeVar("ResultT")


async def _timed(
    timings: dict[str, float], section: str, work: Callable[[], Awaitable[ResultT]]
) -> ResultT:
    """Run one section and record its wall-clock seconds under its name."""
    started = time.perf_counter()
    try:
        return await work()
    finally:
        timings[section] = time.perf_counter() - started
