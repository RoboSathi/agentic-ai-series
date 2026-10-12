"""Stage 1 — drive the agent over a suite and store every transcript.

Generation is the expensive stage and the only one that talks to the
agent, so it is cached twice over:

- a *run* is identified by (suite, agent fingerprint, dataset hash), so
  rerunning an unchanged agent over an unchanged suite finds the run it
  already made and generates nothing;
- a *transcript* is reused across runs whenever the same agent already
  answered the same case — so editing three cases regenerates three.

The fingerprint is a content hash of the working tree and the effective
config, so any change that could alter an answer regenerates by itself;
there is no "fresh" switch to remember. An interrupted run resumes the same
way: finished cases are found, the rest are generated.
"""

import asyncio

from agent_platform.config import get_settings
from agent_platform.evals.core.contracts import (
    AgentUnderTest,
    EvalPlugin,
    GoldenCase,
    Transcript,
    canonical_hash,
)
from agent_platform.evals.core.fingerprint import AgentFingerprint, compute_fingerprint
from agent_platform.evals.core.pipeline.budget import BudgetSnapshot
from agent_platform.evals.core.pipeline.runs import COMPLETED, ERRORED, close_stage, open_run
from agent_platform.evals.core.store import DuckDbEvalStore, RunRecord, StageRecord
from agent_platform.logging import get_logger


class GenerationStage:
    """Runs the agent under test over a suite, reusing whatever it can."""

    def __init__(self, plugin: EvalPlugin, store: DuckDbEvalStore) -> None:
        self._plugin = plugin
        self._store = store
        self._concurrency = get_settings().evals.generation.max_concurrency
        self._logger = get_logger("evals")

    async def run(self, suite: str, cases: list[GoldenCase]) -> RunRecord:
        """Generate (or reuse) a transcript for every case; return the run.

        The plugin must already be prepared (`EvalPlugin.prepare`), outside
        the event loop: some agents load native models that must not start
        inside a running loop.
        """
        fingerprint = compute_fingerprint(self._plugin.fingerprint_inputs())
        run = self._run_for(suite, cases, fingerprint)
        self._store.upsert_cases(cases)
        stage = StageRecord.begin(run.run_id, "generate")
        self._store.start_stage(stage)
        before = BudgetSnapshot.now()
        done = {
            item.case_id
            for item in self._store.transcripts(run.run_id)
            if not item.transcript.error
        }
        pending = [case for case in cases if case.case_id not in done]
        reused = await self._generate(run, pending, fingerprint.value)
        errored = sum(1 for item in self._store.transcripts(run.run_id) if item.transcript.error)
        detail = f"{errored} errored" if errored else ""
        close_stage(
            self._store, stage, before, cases=len(cases), reused=reused + len(done), detail=detail
        )
        self._store.set_run_status(run.run_id, ERRORED if errored else COMPLETED)
        return self._store.get_run(run.run_id) or run

    def _run_for(
        self, suite: str, cases: list[GoldenCase], fingerprint: AgentFingerprint
    ) -> RunRecord:
        dataset_hash = canonical_hash(sorted(case.case_hash for case in cases))
        existing = self._store.find_run(suite, fingerprint.value, dataset_hash)
        if existing is not None:
            self._logger.info("Reusing run for unchanged agent and dataset", run_id=existing.run_id)
            return existing
        return open_run(self._plugin, self._store, suite, fingerprint, dataset_hash, len(cases))

    async def _generate(self, run: RunRecord, cases: list[GoldenCase], fingerprint: str) -> int:
        """Reuse cached transcripts, generate the rest; return how many were reused."""
        reused = 0
        fresh_cases: list[GoldenCase] = []
        for case in cases:
            cached = self._store.reusable_transcript(case.case_hash, fingerprint)
            if cached is None:
                fresh_cases.append(case)
                continue
            self._store.save_transcript(run, case, cached.transcript, reused_from=cached.run_id)
            reused += 1
        agent = self._plugin.agent_under_test()
        gate = asyncio.Semaphore(self._concurrency)

        async def one(case: GoldenCase) -> None:
            async with gate:
                transcript = await self._attempt(agent, case)
            self._store.save_transcript(run, case, transcript)
            self._logger.info(
                "Case generated", case_id=case.case_id, error=transcript.error or None
            )

        await asyncio.gather(*(one(case) for case in fresh_cases))
        return reused

    async def _attempt(self, agent: AgentUnderTest, case: GoldenCase) -> Transcript:
        """One case; an exception becomes an errored transcript, not a crashed run."""
        try:
            return await agent.run(case)
        except Exception as exc:
            self._logger.exception("Case raised during generation", case_id=case.case_id)
            return Transcript(
                suite=case.suite, case_id=case.case_id, error=f"{type(exc).__name__}: {exc}"
            )
