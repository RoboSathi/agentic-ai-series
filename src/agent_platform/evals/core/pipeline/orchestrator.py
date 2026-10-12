"""The eval pipeline: three sections, each its own grader, gate and report.

    oracle     agent runs → the agent's business invariants  → oracle-<run_id>.md
    deepeval   agent runs → LLM-judged quality               → deepeval-<run_id>.md
    promptfoo  red-team attacks against the agent            → promptfoo-<run_id>.md
    intent     the agent's classifier bake-off               → intent-<run_id>.md

The oracle and DeepEval grade the same golden transcripts, so the agent runs
once for both (and not at all while its fingerprint is unchanged). By default
a run covers the agent's core cases, attacks and metrics (`evals.core`);
`full` covers all of them. Each section is timed, so the run ends with where
the time went.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable, Iterator, Sequence
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Literal, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict

from agent_platform.config import get_settings
from agent_platform.evals.core.contracts import (
    EvalPlugin,
    GoldenCase,
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
from agent_platform.evals.core.progress import EvalProgress, NoProgress
from agent_platform.evals.core.reporting import Overview
from agent_platform.evals.core.store import DuckDbEvalStore, RunRecord
from agent_platform.evals.core.store.records import new_run_id
from agent_platform.logging import get_logger

# The sections that grade golden transcripts (each one an eval stage of the same name).
GoldenSection = Literal["oracle", "deepeval"]
# The sections a run is timed by (named for display in `evals.reporting.overview.sections`).
AGENT_SECTION, INTENT_SECTION = "agent", "intent"
ORACLE_SECTION: GoldenSection = "oracle"
DEEPEVAL_SECTION: GoldenSection = "deepeval"
PROMPTFOO_SECTION = "promptfoo"
# The sections that grade the golden set, in the order they run.
GOLDEN_SECTIONS: tuple[GoldenSection, ...] = (ORACLE_SECTION, DEEPEVAL_SECTION)

# Told each golden section's outcome as it finishes (or the error it raised).
SectionSink = Callable[[str, "StageOutcome | Exception"], None]


class NoRunError(LookupError):
    """A report was asked for, but no such run exists."""


class NoIntentBakeoffError(LookupError):
    """`make intent-eval` for an agent that declares no bake-off."""


class StageOutcome(BaseModel):
    """What one section produced: its gate, its report, and the overview to print."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    section: str
    run_id: str
    gate: GateResult
    report: Path
    overview: Overview

    @property
    def failed(self) -> bool:
        return self.gate.verdict == FAIL


class EvalPipeline:
    """Runs an agent's sections and writes their reports."""

    def __init__(
        self,
        plugin: EvalPlugin,
        store: DuckDbEvalStore | None = None,
        progress: EvalProgress | None = None,
    ) -> None:
        self._plugin = plugin
        self._store = store or DuckDbEvalStore()
        self._progress = progress or NoProgress()
        self.clock = SectionClock()
        self._settings = get_settings().evals
        self._logger = get_logger("evals")

    @property
    def store(self) -> DuckDbEvalStore:
        return self._store

    async def golden(
        self,
        sections: Sequence[GoldenSection] = GOLDEN_SECTIONS,
        full: bool = False,
        sink: SectionSink | None = None,
    ) -> list[StageOutcome]:
        """Run the agent over the golden cases once, then each grader asked for, in turn.

        Each section's outcome goes to `sink` as it finishes, so its table prints before the
        next section starts; a section that raises is handed to `sink` and the next still
        runs. Without a sink, an error propagates.
        """
        suite = self._settings.suites.golden
        cases = expand_trials(self._core(self._plugin.load_cases(suite), full))
        generation = GenerationStage(self._plugin, self._store, self._progress)
        run = await self.clock.run(AGENT_SECTION, lambda: generation.run(suite, cases))
        outcomes: list[StageOutcome] = []
        for section in sections:
            try:
                outcome = await self.clock.run(section, partial(self._grade, section, run, full))
            except Exception as exc:
                if sink is None:
                    raise
                sink(section, exc)
                continue
            if outcome is not None:
                outcomes.append(outcome)
                if sink is not None:
                    sink(section, outcome)
        return outcomes

    def security(self, full: bool = False) -> StageOutcome:
        """The agent's Promptfoo red-team suite (reused when nothing changed)."""
        from agent_platform.evals.promptfoo import PromptfooSuite

        with self.clock.measure(PROMPTFOO_SECTION):
            suite = PromptfooSuite(self._plugin, self._store, progress=self._progress, full=full)
            run = suite.run()
        return self._outcome(PROMPTFOO_SECTION, run)

    async def intent(self) -> StageOutcome:
        """The agent's intent classifier bake-off — offline, no provider calls."""
        from agent_platform.evals.intent import BakeoffResult
        from agent_platform.evals.intent import report as intent_report

        if not isinstance(self._plugin, IntentBakeoffPlugin):
            raise NoIntentBakeoffError(f"'{self._plugin.agent_id}' declares no intent bake-off.")
        self._progress.start(INTENT_SECTION, None)
        with self.clock.measure(INTENT_SECTION):
            result = BakeoffResult.model_validate(await self._plugin.intent_bakeoff())
        self._progress.finish(INTENT_SECTION)
        fingerprint = compute_fingerprint(self._plugin.fingerprint_inputs())
        run_id = new_run_id(run_prefix(self._plugin), "intent", fingerprint.value)
        path = intent_report.write(result, run_id)
        gate = GateResult(verdict=PASS, soft_threshold=self._settings.gate.soft_pass_rate)
        return StageOutcome(
            section=INTENT_SECTION,
            run_id=run_id,
            gate=gate,
            report=path,
            overview=intent_report.overview(result, run_id, path),
        )

    def report(self, run_id: str | None = None) -> list[StageOutcome]:
        """Rebuild a run's reports from the store (default: the latest golden run)."""
        run = (
            self._store.get_run(run_id)
            if run_id
            else self._store.latest_run(self._settings.suites.golden)
        )
        if run is None:
            raise NoRunError(f"No eval run {run_id or 'yet'} — `make eval <agent>` records one.")
        if run.suite == self._settings.suites.security:
            return [self._outcome(PROMPTFOO_SECTION, run)]
        graded = {
            row["stage"]
            for row in self._store.query(
                "SELECT DISTINCT stage FROM v_scores WHERE run_id = ?", (run.run_id,)
            )
        }
        return [self._outcome(section, run) for section in GOLDEN_SECTIONS if section in graded]

    async def _grade(
        self, section: GoldenSection, run: RunRecord, full: bool
    ) -> StageOutcome | None:
        """One grader over the run's transcripts, then its report (None: nothing to grade)."""
        if section == ORACLE_SECTION:
            graders: list[Grader] = [self._plugin.oracle()]
        else:
            metrics = self._core_metrics(full)
            if not metrics:
                return None
            graders = [self._deepeval(metrics)]
        by_id = self._store.cases_for_run(run.run_id)
        await GradingStage(self._store, self._progress).run(run.run_id, section, graders, by_id)
        return self._outcome(section, run)

    def _outcome(self, section: str, run: RunRecord) -> StageOutcome:
        """Write the section's report and build its overview."""
        report = _report_for(section, self._store)
        path = report.write(run)
        return StageOutcome(
            section=section,
            run_id=run.run_id,
            gate=evaluate_gate(self._store, run.run_id, section),
            report=path,
            overview=report.overview(run, path),
        )

    def _core(self, cases: list[GoldenCase], full: bool) -> list[GoldenCase]:
        """The agent's core cases, unless `full` — or every case, if it marked none core."""
        tag = self._settings.core.tag
        core = [case for case in cases if tag in case.tags]
        return cases if full or not core else core

    def _core_metrics(self, full: bool) -> list[QualityMetricSpec]:
        metrics = self._plugin.quality_metrics()
        core = [metric for metric in metrics if metric.core]
        return metrics if full or not core else core

    def _deepeval(self, metrics: list[QualityMetricSpec]) -> Grader:
        # Imported here: DeepEval is the optional `evals` extra.
        from agent_platform.evals.deepeval import DeepEvalGrader

        return DeepEvalGrader(metrics, self._plugin.conversation_frame)


class _SectionReport(Protocol):
    def write(self, run: RunRecord) -> Path: ...

    def overview(self, run: RunRecord, report: Path) -> Overview: ...


def _report_for(section: str, store: DuckDbEvalStore) -> _SectionReport:
    """Each section's own report (imported late: DeepEval is the optional `evals` extra)."""
    if section == ORACLE_SECTION:
        from agent_platform.evals.oracle.report import OracleReport

        return OracleReport(store)
    if section == DEEPEVAL_SECTION:
        from agent_platform.evals.deepeval.report import DeepEvalReport

        return DeepEvalReport(store)
    from agent_platform.evals.promptfoo.report import PromptfooReport

    return PromptfooReport(store)


ResultT = TypeVar("ResultT")


class SectionClock:
    """How long each section of a run took — kept as it goes, so an interrupt can show it.

    `unfinished` names the section that was running when Ctrl+C stopped the run.
    """

    def __init__(self) -> None:
        self.elapsed: dict[str, float] = {}
        self.unfinished: str | None = None

    async def run(self, section: str, work: Callable[[], Awaitable[ResultT]]) -> ResultT:
        with self.measure(section):
            return await work()

    @contextmanager
    def measure(self, section: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        except (KeyboardInterrupt, asyncio.CancelledError):
            # Stopped, not broken: a section that raised is reported as an error instead.
            self.unfinished = section
            raise
        finally:
            self.elapsed[section] = time.perf_counter() - started

    def of(self, *sections: str) -> dict[str, float]:
        """The named sections' times, in run order."""
        return {name: seconds for name, seconds in self.elapsed.items() if name in sections}
