"""The security suite: Promptfoo runs the attacks, the eval store keeps the results.

Promptfoo is a Node tool, run through `npx` at a pinned version — never a
Python dependency — the standard way: `promptfoo eval -c promptfooconfig.yaml`.
The agent's own config declares everything Promptfoo needs (its provider,
its grader, its attacks and assertions); the platform supplies the two
Python entry points those name (`agent_provider`, `grader_provider`), runs
the command, and ingests Promptfoo's JSON into DuckDB, where the report and
the gate read it.

Each attack's metadata names its place in Promptfoo's red-team taxonomy —
the plugin (`hijacking`, `pii:direct`, `bola`, …), the strategy that
delivers it (`jailbreak`, `base64`, `crescendo`, …) and its severity — and
the report groups by them.

Reuse follows the same rule as generation: a completed security run for
this exact agent fingerprint and this exact attack set is not re-run.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from agent_platform.config import get_settings, resolve_path
from agent_platform.config.settings import EvalPromptfooSettings
from agent_platform.evals.core.contracts import (
    EvalPlugin,
    GoldenCase,
    Score,
    Transcript,
    canonical_hash,
)
from agent_platform.evals.core.fingerprint import compute_fingerprint
from agent_platform.evals.core.pipeline.budget import BudgetSnapshot
from agent_platform.evals.core.pipeline.runs import COMPLETED, ERRORED, close_stage, open_run
from agent_platform.evals.core.progress import EvalProgress, NoProgress
from agent_platform.evals.core.store import DuckDbEvalStore, RunRecord, StageRecord
from agent_platform.evals.promptfoo.agent_provider import case_from_vars
from agent_platform.logging import get_logger

_PYTHON_ENV = "PROMPTFOO_PYTHON"
_MODEL_GRADED = frozenset({"llm-rubric", "model-graded-closedqa", "g-eval"})
_RUN_METRIC = "run_completed"
_SUITE_SUFFIXES = (".yaml", ".yml", ".py", ".json")
_SEVERITY_TAG = "severity:"
_TESTS_KEY = "tests"
_METADATA_KEY = "metadata"
_FILE_SCHEME = "file://"
# The progress section this suite reports under (named in `evals.reporting.overview.sections`).
SECTION = "promptfoo"
_STRATEGY_TAG = "strategy:"


class PromptfooRunError(RuntimeError):
    """Promptfoo exited without writing results."""


class PromptfooSuite:
    """Runs (or reuses) the agent's Promptfoo red-team suite."""

    def __init__(
        self,
        plugin: EvalPlugin,
        store: DuckDbEvalStore,
        settings: EvalPromptfooSettings | None = None,
        progress: EvalProgress | None = None,
        full: bool = False,
    ) -> None:
        """`full`: every attack; otherwise only those the agent marked core (`evals.core`)."""
        self._plugin = plugin
        self._store = store
        self._progress = progress or NoProgress()
        self._settings = settings or get_settings().evals.promptfoo
        self._suite = get_settings().evals.suites.security
        self._config = resolve_path(plugin.security_config())
        core = get_settings().evals.core
        only = (core.attack_metadata_key, core.tag)
        # An agent that marked no attack core runs them all.
        self._only = None if full or not attack_count(self._config, only) else only
        self._logger = get_logger("evals")

    def run(self) -> RunRecord:
        fingerprint = compute_fingerprint(self._plugin.fingerprint_inputs())
        # The core subset and the full set are different attack sets: different runs.
        attacks = canonical_hash([suite_hash(self._config.parent), self._only])
        total = attack_count(self._config, self._only)
        self._progress.start(SECTION, total)
        existing = self._store.find_run(self._suite, fingerprint.value, attacks)
        if existing is not None and existing.status == COMPLETED:
            self._logger.info(
                "Attack set unchanged for this agent; reusing", run_id=existing.run_id
            )
            self._progress.advance(SECTION, total or 0)
            self._progress.finish(SECTION)
            return existing
        run = open_run(self._plugin, self._store, self._suite, fingerprint, attacks, 0)
        stage = StageRecord.begin(run.run_id, "promptfoo", attacks)
        self._store.start_stage(stage)
        before = BudgetSnapshot.now()
        results = self.ingest(self._execute(run), run, stage)
        self._finish(run, stage, before, results)
        self._progress.finish(SECTION)
        return self._store.get_run(run.run_id) or run

    def command(self, output: Path) -> list[str]:
        """The exact Promptfoo invocation (public so a report or test can show it)."""
        return [
            *self._settings.runner,
            self._settings.package,
            "eval",
            "-c",
            str(self._config),
            "-j",
            str(self._settings.max_concurrency),
            "-o",
            str(output),
            *self._settings.extra_args,
            *(["--filter-metadata", f"{self._only[0]}={self._only[1]}"] if self._only else []),
        ]

    def ingest(self, output: Path, run: RunRecord, stage: StageRecord) -> list[dict[str, Any]]:
        """Store every Promptfoo result as a transcript plus hard scores."""
        results = latest_per_case(json.loads(output.read_text())["results"]["results"])
        for result in results:
            case, transcript = self._case_and_transcript(result)
            self._store.upsert_cases([case])
            stored = self._store.save_transcript(run, case, transcript)
            self._store.save_scores(stage, stored, _scores(case.case_id, result), reused=False)
        return list(results)

    def _execute(self, run: RunRecord) -> Path:
        folder = self._settings.resolved_output_dir
        folder.mkdir(parents=True, exist_ok=True)
        output = folder / f"{run.run_id}.json"
        ticks = folder / f"{run.run_id}.progress"
        ticks.unlink(missing_ok=True)
        command = self.command(output)
        env = {
            **os.environ,
            **self._settings.env,
            _PYTHON_ENV: sys.executable,
            self._settings.progress_env: str(ticks),
        }
        self._logger.info("Running Promptfoo security suite", command=command)
        with subprocess.Popen(
            command,
            cwd=self._config.parent,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ) as process:
            try:
                stdout, stderr = self._wait(process, ticks)
            finally:
                if process.poll() is None:
                    process.terminate()
        self._logger.info(
            "Promptfoo finished", exit_code=process.returncode, stdout=stdout, stderr=stderr
        )
        if not output.exists():
            raise PromptfooRunError(
                f"Promptfoo exited {process.returncode} without results: {stderr.strip()}"
            )
        return output

    def _wait(self, process: "subprocess.Popen[str]", ticks: Path) -> tuple[str, str]:
        """Wait for Promptfoo, reporting each attack the agent finished (one line per attack)."""
        while True:
            try:
                return process.communicate(timeout=self._settings.progress_poll_seconds)
            except subprocess.TimeoutExpired:
                self._progress.advance(SECTION, _lines(ticks))

    def _case_and_transcript(self, result: dict[str, Any]) -> tuple[GoldenCase, Transcript]:
        """The attack as a case — grouped by its red-team plugin — and what the agent did."""
        metadata = (result.get("response") or {}).get("metadata") or {}
        prompt = str((result.get("prompt") or {}).get("raw", ""))
        case = (
            GoldenCase.model_validate(metadata["case"])
            if "case" in metadata
            else case_from_vars(prompt, result.get("vars") or {})
        )
        case = self._taxonomy(case, (result.get("testCase") or {}).get("metadata") or {})
        if "transcript" in metadata:
            return case, Transcript.model_validate(metadata["transcript"])
        error = str(result.get("error") or "provider returned no transcript")
        return case, Transcript(suite=case.suite, case_id=case.case_id, error=error)

    def _taxonomy(self, case: GoldenCase, metadata: dict[str, Any]) -> GoldenCase:
        plugin = metadata.get(self._settings.plugin_key)
        strategy = metadata.get(self._settings.strategy_key)
        severity = metadata.get(self._settings.severity_key)
        tags = [
            *case.tags,
            *([f"{_STRATEGY_TAG}{strategy}"] if strategy else []),
            *([f"{_SEVERITY_TAG}{severity}"] if severity else []),
        ]
        return case.model_copy(update={"category": str(plugin or case.category), "tags": tags})

    def _finish(
        self,
        run: RunRecord,
        stage: StageRecord,
        before: BudgetSnapshot,
        results: list[dict[str, Any]],
    ) -> None:
        # The agent ran in Promptfoo's workers, whose calls this process's limiter never saw.
        requests = sum(
            int((result.get("response") or {}).get("tokenUsage", {}).get("numRequests") or 0)
            for result in results
        )
        errored = sum(1 for result in results if result.get("error"))
        detail = f"{errored} errored" if errored else ""
        close_stage(
            self._store, stage, before, cases=len(results), detail=detail, requests=requests
        )
        self._store.execute(
            "UPDATE eval_runs SET case_count = ? WHERE run_id = ?", (len(results), run.run_id)
        )
        self._store.set_run_status(run.run_id, ERRORED if errored else COMPLETED)


def attack_count(config: Path, only: tuple[str, str] | None = None) -> int | None:
    """How many attacks the config declares (its `tests`, inline or in one file), if knowable.

    `only` (`key`, `value`): count just the attacks whose metadata has it — what Promptfoo's
    `--filter-metadata key=value` will run.
    """
    tests = (yaml.safe_load(config.read_text(encoding="utf-8")) or {}).get(_TESTS_KEY)
    if isinstance(tests, str) and tests.startswith(_FILE_SCHEME):
        source = config.parent / tests.removeprefix(_FILE_SCHEME)
        tests = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(tests, list):
        return None
    if only is None:
        return len(tests)
    key, value = only
    return sum(1 for test in tests if str((test.get(_METADATA_KEY) or {}).get(key)) == value)


def _lines(path: Path) -> int:
    try:
        return len(path.read_text(encoding="utf-8").splitlines())
    except FileNotFoundError:
        return 0


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
    return canonical_hash([(path.name, _digest(path)) for path in files])


def _digest(path: Path) -> str:
    """A file's content hash, read as bytes: an attack set may hold any encoding."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def severity_of(case: GoldenCase) -> str:
    """The attack's red-team severity, from its tags ("" when it declared none)."""
    return _tag(case.tags, _SEVERITY_TAG)


def strategy_of(case: GoldenCase) -> str:
    """How the attack was delivered, from its tags ("" when it declared none)."""
    return _tag(case.tags, _STRATEGY_TAG)


def _tag(tags: list[str], prefix: str) -> str:
    return next((tag.removeprefix(prefix) for tag in tags if tag.startswith(prefix)), "")


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
