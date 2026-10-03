"""The security suite: Promptfoo runs the attacks, the eval store keeps the results.

Promptfoo is a Node tool, run through `npx` at a pinned version — never a
Python dependency. The agent's own Promptfoo config declares the attacks
and assertions; this module supplies the two things that are the same for
every agent — the provider that runs the agent under test and the grader
that routes `llm-rubric` to the eval judge — and ingests Promptfoo's JSON
output into DuckDB, where the reports and the gate read it.

Reuse follows the same rule as generation: a completed security run for
this exact agent fingerprint and this exact attack set is not re-run.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from agent_platform.config import get_settings, resolve_path
from agent_platform.config.settings import EvalPromptfooSettings
from agent_platform.evals.adapters.promptfoo_agent_provider import case_from_vars
from agent_platform.evals.contracts import (
    EvalPlugin,
    GoldenCase,
    Score,
    Transcript,
    canonical_hash,
)
from agent_platform.evals.fingerprint import compute_fingerprint
from agent_platform.evals.pipeline.budget import current_limit
from agent_platform.evals.store import DuckDbEvalStore, RunRecord, StageRecord
from agent_platform.evals.store.records import new_run_id
from agent_platform.logging import get_logger
from agent_platform.observability.types import utc_now

_ADAPTERS = Path(__file__).resolve().parent
AGENT_PROVIDER = _ADAPTERS / "promptfoo_agent_provider.py"
JUDGE_PROVIDER = _ADAPTERS / "promptfoo_judge_provider.py"
_PYTHON_ENV = "PROMPTFOO_PYTHON"
_FILE_SCHEME = "file://"
_MODEL_GRADED = frozenset({"llm-rubric", "model-graded-closedqa", "g-eval"})
_RUN_METRIC = "run_completed"
_SUITE_SUFFIXES = (".yaml", ".yml", ".py", ".json")


class PromptfooRunError(RuntimeError):
    """Promptfoo exited without writing results."""


class PromptfooSuite:
    """Runs (or reuses) the agent's Promptfoo security suite."""

    def __init__(
        self,
        plugin: EvalPlugin,
        store: DuckDbEvalStore,
        settings: EvalPromptfooSettings | None = None,
    ) -> None:
        self._plugin = plugin
        self._store = store
        self._settings = settings or get_settings().evals.promptfoo
        self._config = resolve_path(plugin.security_config())
        self._logger = get_logger("evals")

    def run(self, *, fresh: bool = False) -> RunRecord:
        fingerprint = compute_fingerprint(self._plugin.fingerprint_inputs())
        dataset = suite_hash(self._config.parent)
        existing = self._store.find_run(self._settings.suite, fingerprint.value, dataset)
        if existing is not None and existing.status == "completed" and not fresh:
            self._logger.info(
                "Security suite unchanged for this agent; reusing", run_id=existing.run_id
            )
            return existing
        run = self._new_run(
            fingerprint.value,
            fingerprint.parts,
            dataset,
            fingerprint.git_sha,
            fingerprint.git_dirty,
        )
        stage = StageRecord.begin(run.run_id, "security", dataset)
        self._store.start_stage(stage)
        output = self._execute(run)
        ingested = self.ingest(output, run, stage)
        self._finish(run, stage, ingested)
        return self._store.get_run(run.run_id) or run

    def command(self, output: Path, providers: Path) -> list[str]:
        """The exact Promptfoo invocation (public so a report or test can show it)."""
        return [
            *self._settings.runner,
            self._settings.package,
            "eval",
            "-c",
            str(self._config),
            "-r",
            f"{_FILE_SCHEME}{providers}",
            "--grader",
            f"{_FILE_SCHEME}{JUDGE_PROVIDER}",
            "-j",
            str(self._settings.max_concurrency),
            "-o",
            str(output),
            *self._settings.extra_args,
        ]

    def providers_file(self, folder: Path, run_id: str) -> Path:
        """The agent-under-test provider, with the worker timeout and pool size from config.

        A bare `-r file://provider.py` cannot carry provider config, and
        Promptfoo's default worker timeout is shorter than a multi-turn
        attack behind the shared rate limit.
        """
        path = folder / f"{run_id}.providers.yaml"
        spec = [
            {
                "id": f"{self._settings.provider_scheme}{AGENT_PROVIDER}",
                "label": self._plugin.agent_id,
                "config": {
                    "timeout": self._settings.provider_timeout_ms,
                    "workers": self._settings.max_concurrency,
                },
            }
        ]
        path.write_text(yaml.safe_dump(spec, sort_keys=False))
        return path

    def ingest(self, output: Path, run: RunRecord, stage: StageRecord) -> list[dict[str, Any]]:
        """Store every Promptfoo result as a transcript plus hard scores."""
        results = latest_per_case(json.loads(output.read_text())["results"]["results"])
        for result in results:
            case, transcript = _case_and_transcript(result)
            self._store.upsert_cases([case])
            stored = self._store.save_transcript(run, case, transcript)
            self._store.save_scores(stage, stored, _scores(case.case_id, result), reused=False)
        return list(results)

    def _execute(self, run: RunRecord) -> Path:
        folder = self._settings.resolved_output_dir
        folder.mkdir(parents=True, exist_ok=True)
        output = folder / f"{run.run_id}.json"
        command = self.command(output, self.providers_file(folder, run.run_id))
        env = {**os.environ, **self._settings.env, _PYTHON_ENV: sys.executable}
        self._logger.info("Running Promptfoo security suite", command=command)
        completed = subprocess.run(
            command, cwd=self._config.parent, env=env, text=True, capture_output=True
        )
        self._logger.info(
            "Promptfoo finished",
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
        if not output.exists():
            raise PromptfooRunError(
                f"Promptfoo exited {completed.returncode} without results: {completed.stderr.strip()}"
            )
        return output

    def _new_run(
        self, fingerprint: str, parts: dict[str, str], dataset: str, git_sha: str, dirty: bool
    ) -> RunRecord:
        suite = self._settings.suite
        run = RunRecord(
            run_id=new_run_id(suite, fingerprint),
            suite=suite,
            agent_id=self._plugin.agent_id,
            agent_fingerprint=fingerprint,
            fingerprint_parts=parts,
            dataset_hash=dataset,
            case_count=0,
            git_sha=git_sha,
            git_dirty=dirty,
            model=self._plugin.fingerprint_inputs().get("model", ""),
            judge_model=get_settings().evals.judge.model,
            rate_limit=current_limit(),
            started_at=utc_now(),
            status="running",
        )
        self._store.create_run(run)
        return run

    def _finish(self, run: RunRecord, stage: StageRecord, results: list[dict[str, Any]]) -> None:
        requests = sum(
            int((result.get("response") or {}).get("tokenUsage", {}).get("numRequests") or 0)
            for result in results
        )
        errored = sum(1 for result in results if result.get("error"))
        self._store.finish_stage(
            stage.model_copy(
                update={
                    "finished_at": utc_now(),
                    "status": "completed",
                    "cases": len(results),
                    "provider_requests": requests,
                    "detail": f"{errored} errored" if errored else "",
                }
            )
        )
        self._store.execute(
            "UPDATE eval_runs SET case_count = ? WHERE run_id = ?", (len(results), run.run_id)
        )
        self._store.set_run_status(run.run_id, "errored" if errored else "completed")


def latest_per_case(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One result per attack: the last clean attempt, else the last attempt.

    Promptfoo can run a test more than once (a retried worker); a repeated
    attack must not be scored twice.
    """
    chosen: dict[str, dict[str, Any]] = {}
    for result in results:
        case_id = str((result.get("vars") or {}).get("case_id", ""))
        kept = chosen.get(case_id)
        if kept is None or not result.get("error") or kept.get("error"):
            chosen[case_id] = result
    return list(chosen.values())


def suite_hash(folder: Path) -> str:
    """Identity of the attack set: every config, attack and assertion file's contents."""
    files = sorted(path for path in folder.rglob("*") if path.suffix in _SUITE_SUFFIXES)
    return canonical_hash([(path.name, path.read_text()) for path in files])


def _case_and_transcript(result: dict[str, Any]) -> tuple[GoldenCase, Transcript]:
    metadata = (result.get("response") or {}).get("metadata") or {}
    test_vars = result.get("vars") or {}
    prompt = str((result.get("prompt") or {}).get("raw", ""))
    case = (
        GoldenCase.model_validate(metadata["case"])
        if "case" in metadata
        else case_from_vars(prompt, test_vars)
    )
    if "transcript" in metadata:
        return case, Transcript.model_validate(metadata["transcript"])
    error = str(result.get("error") or "provider returned no transcript")
    return case, Transcript(suite=case.suite, case_id=case.case_id, error=error)


def _scores(case_id: str, result: dict[str, Any]) -> list[Score]:
    grading = result.get("gradingResult") or {}
    components = grading.get("componentResults") or []
    if not components:
        return [_score(case_id, _RUN_METRIC, "python", False, 0.0, str(result.get("error") or ""))]
    return [
        _score(
            case_id,
            str(item["assertion"].get("metric") or item["assertion"].get("type")),
            str(item["assertion"].get("type")),
            bool(item.get("pass")),
            float(item.get("score") or 0.0),
            str(item.get("reason") or ""),
        )
        for item in components
    ]


def _score(case_id: str, metric: str, kind: str, passed: bool, value: float, why: str) -> Score:
    return Score(
        case_id=case_id,
        metric=metric,
        grader="promptfoo",
        kind="promptfoo",
        severity="hard",
        value=value,
        passed=passed,
        threshold=1.0,
        tier="calibrated_llm_judge" if kind in _MODEL_GRADED else "deterministic_rule",
        rationale=why,
    )
