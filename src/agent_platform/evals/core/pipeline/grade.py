"""Grading stages — score a run's transcripts with one kind of grader.

Every grading stage has the same shape: for each transcript, reuse the
scores this exact grader already gave this exact transcript, or grade it
now. "Exact grader" is the grader fingerprint — names, versions, and
whatever the grader says changes its judgement (judge model, metric spec)
— so editing a metric regrades, and rerunning an unchanged one costs nothing.

A grader that scores many transcripts in one call (`BatchGrader`, DeepEval) is
handed them a batch at a time; every other grader, one transcript at a time.
"""

import asyncio

from agent_platform.config import get_settings
from agent_platform.evals.core.contracts import (
    BatchGrader,
    GoldenCase,
    Grader,
    Score,
    StageName,
    Transcript,
    canonical_hash,
)
from agent_platform.evals.core.pipeline.budget import BudgetSnapshot
from agent_platform.evals.core.pipeline.runs import close_stage
from agent_platform.evals.core.progress import EvalProgress, NoProgress
from agent_platform.evals.core.store import DuckDbEvalStore, StageRecord, StoredTranscript
from agent_platform.logging import get_logger

# A grader that raised leaves a score under this suffix. Such a score is
# never reused: the next run must try again, not replay the failure.
ERROR_METRIC_SUFFIX = ".error"


def grader_fingerprint(graders: list[Grader]) -> str:
    """Identity of a set of graders: what must match for scores to be reused."""
    return canonical_hash(sorted((grader.name, grader.version) for grader in graders))


class GradingStage:
    """Applies graders to every transcript of a run."""

    def __init__(self, store: DuckDbEvalStore, progress: EvalProgress | None = None) -> None:
        self._store = store
        self._progress = progress or NoProgress()
        self._concurrency = get_settings().evals.generation.max_concurrency
        self._logger = get_logger("evals")

    async def run(
        self,
        run_id: str,
        stage_name: StageName,
        graders: list[Grader],
        cases: dict[str, GoldenCase],
    ) -> StageRecord:
        """Score (or re-use scores for) every transcript in the run."""
        stage = StageRecord.begin(run_id, stage_name, grader_fingerprint(graders))
        self._store.start_stage(stage)
        before = BudgetSnapshot.now()
        stored = [item for item in self._store.transcripts(run_id) if item.case_id in cases]
        self._progress.start(stage_name, len(stored))
        pending = [item for item in stored if not self._reused(stage, item)]
        done = len(stored) - len(pending)
        self._progress.advance(stage_name, done)
        batch = graders[0] if len(graders) == 1 and isinstance(graders[0], BatchGrader) else None
        if batch is not None:
            await self._in_batches(stage, batch, pending, cases, done)
        else:
            await self._one_by_one(stage, graders, pending, cases, done)
        self._progress.finish(stage_name)
        return close_stage(self._store, stage, before, cases=len(stored), reused=done)

    def _reused(self, stage: StageRecord, item: StoredTranscript) -> bool:
        """Save the scores this exact grader already gave this transcript, if any."""
        cached = self._store.reusable_scores(
            item.transcript_hash, stage.stage, stage.grader_fingerprint
        )
        if not cached or any(s.metric.endswith(ERROR_METRIC_SUFFIX) for s in cached):
            return False
        self._store.save_scores(stage, item, cached, reused=True)
        return True

    async def _one_by_one(
        self,
        stage: StageRecord,
        graders: list[Grader],
        pending: list[StoredTranscript],
        cases: dict[str, GoldenCase],
        done: int,
    ) -> None:
        gate = asyncio.Semaphore(self._concurrency)
        finished = done

        async def one(item: StoredTranscript) -> None:
            nonlocal finished
            async with gate:
                scores: list[Score] = []
                for grader in graders:
                    scores.extend(await self._grade(grader, cases[item.case_id], item))
            self._store.save_scores(stage, item, scores, reused=False)
            finished += 1
            self._progress.advance(stage.stage, finished)

        await asyncio.gather(*(one(item) for item in pending))

    async def _in_batches(
        self,
        stage: StageRecord,
        grader: BatchGrader,
        pending: list[StoredTranscript],
        cases: dict[str, GoldenCase],
        done: int,
    ) -> None:
        size = grader.batch_size
        for start in range(0, len(pending), size):
            chunk = pending[start : start + size]
            items = [(cases[item.case_id], item.transcript) for item in chunk]
            for item, scores in zip(chunk, await self._grade_batch(grader, items), strict=True):
                self._store.save_scores(stage, item, scores, reused=False)
            done += len(chunk)
            self._progress.advance(stage.stage, done)

    async def _grade_batch(
        self, grader: BatchGrader, items: list[tuple[GoldenCase, Transcript]]
    ) -> list[list[Score]]:
        """One batch; a batch that raises records an error score per case, never a crash."""
        try:
            return await grader.grade_batch(items)
        except Exception as exc:
            self._logger.exception("Grader raised", grader=grader.name, cases=len(items))
            return [[_error_score(grader, case.case_id, exc)] for case, _ in items]

    async def _grade(self, grader: Grader, case: GoldenCase, item: StoredTranscript) -> list[Score]:
        """One grader; a grader that raises records an error score, never a crash."""
        try:
            return await grader.grade(case, item.transcript)
        except Exception as exc:
            self._logger.exception("Grader raised", grader=grader.name, case_id=case.case_id)
            return [_error_score(grader, case.case_id, exc)]


def _error_score(grader: Grader, case_id: str, exc: Exception) -> Score:
    return Score(
        case_id=case_id,
        metric=f"{grader.name}{ERROR_METRIC_SUFFIX}",
        grader=grader.name,
        kind=grader.kind,
        severity="soft",
        value=0.0,
        passed=False,
        tier="human_adjudication",
        rationale=f"{type(exc).__name__}: {exc}",
    )
