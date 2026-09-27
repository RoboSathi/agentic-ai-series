"""The eval pipeline: one method per make target, each ending in its report.

    generate → oracle → security → judge → deepeval

Cheapest and most severe first. Generation must come first (everything
reads its transcripts); the oracle is free and catches the defects that
make quality scores moot, so by default a broken invariant stops the
pipeline before any quota is spent on the judge; security is next
because a successful attack also blocks a release; the soft, expensive
quality graders run last.
"""

from pathlib import Path

from agentic_commons.config import get_settings
from agentic_commons.evals.contracts import EvalPlugin, GoldenCase, Grader, StageName
from agentic_commons.evals.graders import RubricJudge
from agentic_commons.evals.pipeline.gate import FAIL, GateResult, evaluate_gate
from agentic_commons.evals.pipeline.generate import GenerationStage
from agentic_commons.evals.pipeline.grade import GradingStage
from agentic_commons.evals.reporting import EvalReporter
from agentic_commons.evals.store import DuckDbEvalStore, RunRecord
from agentic_commons.logging import get_logger


class NoRunError(LookupError):
    """A stage was asked to grade, but no run exists to grade."""


class EvalPipeline:
    """Runs stages against a plugin's agent and writes their reports."""

    def __init__(
        self,
        plugin: EvalPlugin,
        store: DuckDbEvalStore | None = None,
        reporter: EvalReporter | None = None,
    ) -> None:
        self._plugin = plugin
        self._store = store or DuckDbEvalStore()
        self._reporter = reporter or EvalReporter(self._store)
        self._settings = get_settings().evals
        self._logger = get_logger("evals")

    @property
    def store(self) -> DuckDbEvalStore:
        return self._store

    async def generate(
        self, suite: str, cases: list[GoldenCase], *, fresh: bool = False
    ) -> RunRecord:
        run, _ = await GenerationStage(self._plugin, self._store).run(suite, cases, fresh=fresh)
        self._reporter.write(run.run_id)
        return run

    async def grade(self, stage: StageName, run_id: str, *, fresh: bool = False) -> GateResult:
        """Score a run with one stage's graders; return the run's gate afterwards."""
        run = self.run_or_latest(run_id)
        cases = self._store.cases_for_run(run.run_id)
        graders = self._graders(stage, run)
        await GradingStage(self._store).run(run.run_id, stage, graders, cases, fresh=fresh)
        self._reporter.write(run.run_id)
        return evaluate_gate(self._store, run.run_id, run.suite)

    def should_stop(self, gate: GateResult) -> bool:
        """After the oracle: stop before spending quota when an invariant broke."""
        return (
            self._settings.gate.stop_on_invariant_failure
            and gate.verdict == FAIL
            and gate.hard_failed_cases > 0
        )

    def report(self, run_id: str | None = None) -> list[Path]:
        return self._reporter.write(self.run_or_latest(run_id).run_id)

    def diff(self, base: str, head: str) -> Path:
        return self._reporter.write_diff(base, head)

    def run_or_latest(self, run_id: str | None, suite: str | None = None) -> RunRecord:
        run = (
            self._store.get_run(run_id)
            if run_id
            else self._store.latest_run(suite or self._settings.suites[0])
        )
        if run is None:
            raise NoRunError(
                f"No eval run {'`' + run_id + '`' if run_id else 'yet'} — run `make eval-generate` first."
            )
        return run

    def _graders(self, stage: StageName, run: RunRecord) -> list[Grader]:
        if stage == "oracle":
            return [self._plugin.oracle()]
        if stage == "judge":
            return [RubricJudge(self._plugin.judge_rubric(run.suite))]
        if stage == "deepeval":
            from agentic_commons.evals.adapters.deepeval_adapter import deepeval_graders

            return deepeval_graders(self._plugin.quality_metrics())
        raise ValueError(f"'{stage}' is not a grading stage")
