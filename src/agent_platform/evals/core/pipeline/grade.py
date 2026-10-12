"""Grading stages — score a run's transcripts with one kind of grader.

Every grading stage has the same shape: for each transcript, reuse the
scores this exact grader already gave this exact transcript, or grade it
now. "Exact grader" is the grader fingerprint — names, versions, and
whatever the grader says changes its judgement (judge model, metric spec)
— so editing a metric regrades, and rerunning an unchanged one costs nothing.
"""

import asyncio

from agent_platform.config import get_settings
from agent_platform.evals.core.contracts import GoldenCase, Grader, Score, StageName, canonical_hash
from agent_platform.evals.core.pipeline.budget import BudgetSnapshot
from agent_platform.evals.core.pipeline.runs import close_stage
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

    def __init__(self, store: DuckDbEvalStore) -> None:
        self._store = store
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
        gate = asyncio.Semaphore(self._concurrency)

        async def one(item: StoredTranscript) -> bool:
            async with gate:
                return await self._score(stage, item, cases[item.case_id], graders)

        reused = await asyncio.gather(*(one(item) for item in stored))
        return close_stage(self._store, stage, before, cases=len(stored), reused=sum(reused))

    async def _score(
        self, stage: StageRecord, item: StoredTranscript, case: GoldenCase, graders: list[Grader]
    ) -> bool:
        """Grade one transcript; return whether cached scores were reused."""
        cached = self._store.reusable_scores(
            item.transcript_hash, stage.stage, stage.grader_fingerprint
        )
        if cached and not any(s.metric.endswith(ERROR_METRIC_SUFFIX) for s in cached):
            self._store.save_scores(stage, item, cached, reused=True)
            return True
        scores: list[Score] = []
        for grader in graders:
            scores.extend(await self._grade(grader, case, item))
        self._store.save_scores(stage, item, scores, reused=False)
        return False

    async def _grade(self, grader: Grader, case: GoldenCase, item: StoredTranscript) -> list[Score]:
        """One grader; a grader that raises records an error score, never a crash."""
        try:
            return await grader.grade(case, item.transcript)
        except Exception as exc:
            self._logger.exception("Grader raised", grader=grader.name, case_id=case.case_id)
            return [
                Score(
                    case_id=case.case_id,
                    metric=f"{grader.name}{ERROR_METRIC_SUFFIX}",
                    grader=grader.name,
                    kind=grader.kind,
                    severity="soft",
                    value=0.0,
                    passed=False,
                    tier="human_adjudication",
                    rationale=f"{type(exc).__name__}: {exc}",
                )
            ]
