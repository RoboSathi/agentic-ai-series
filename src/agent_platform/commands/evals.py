"""`/eval` — the eval store, read from the console (LLD §18, §20).

A view, not a runner: evals spend provider quota for minutes, which is a
`make eval` job, not something to start from a prompt line. This reads
the same DuckDB store and views the Markdown reports are built from, so
the console and the reports can never disagree.
"""

import asyncio
from typing import Any

from agent_platform.commands.types import CommandError, CommandResult
from agent_platform.config import get_settings
from agent_platform.evals.core.pipeline.gate import evaluate_gate
from agent_platform.evals.core.store import DuckDbEvalStore

_SUBCOMMANDS = ("runs", "report", "case")


class EvalCommand:
    """Recent eval runs, one run's scores, or one case."""

    group = "Evals"
    name = "eval"
    summary = "Eval runs, reports and cases (run them with `make eval`)"
    usage = "/eval [runs | report [run] | case <case_id> [run]]"

    def __init__(self, store: DuckDbEvalStore | None = None) -> None:
        self._store = store

    def argument_completions(self, args: list[str]) -> list[str]:
        """The subcommands, in first position only."""
        return [] if args else list(_SUBCOMMANDS)

    async def run(self, args: list[str]) -> CommandResult:
        action, rest = (args[0], args[1:]) if args else ("runs", [])
        if action not in _SUBCOMMANDS:
            raise CommandError(f"Unknown /eval action '{action}'")
        return await asyncio.to_thread(self._read, action, rest)

    def _read(self, action: str, rest: list[str]) -> CommandResult:
        store = self._store or DuckDbEvalStore()
        if action == "runs":
            return _runs(store)
        if action == "report":
            return _report(store, rest[0] if rest else None)
        if not rest:
            raise CommandError("Name a case: /eval case <case_id> [run]")
        return _case(store, rest[0], rest[1] if len(rest) > 1 else None)


def _runs(store: DuckDbEvalStore) -> CommandResult:
    limit = get_settings().commands.default_limit
    rows = store.query(
        "SELECT run_id, suite, status, case_count, hard_failed_cases, soft_pass_rate "
        "FROM v_run_summary ORDER BY started_at DESC LIMIT ?",
        (limit,),
    )
    result = CommandResult(title=f"Eval runs — {len(rows)} most recent")
    result.headers = ["run", "suite", "status", "cases", "hard-failed cases", "soft pass rate"]
    for row in rows:
        result.row(
            *(_shown(value) for value in row.values()),
            style="red" if row["hard_failed_cases"] else "",
        )
    if not rows:
        result.line("No eval runs yet — `make eval` records one.")
    return result


def _report(store: DuckDbEvalStore, run_id: str | None) -> CommandResult:
    run = store.get_run(run_id) if run_id else store.latest_run()
    if run is None:
        raise CommandError("No such eval run — /eval runs lists them.")
    gate = evaluate_gate(store, run.run_id)
    result = CommandResult(title=f"{run.run_id} — {gate.verdict}")
    result.headers = ["stage", "metric", "severity", "passed", "mean"]
    stages: list[str] = []
    for row in store.query(
        "SELECT stage, metric, severity, passed, cases, mean_value FROM v_metric_pass_rates "
        "WHERE run_id = ? ORDER BY stage, severity, metric",
        (run.run_id,),
    ):
        failing = row["passed"] < row["cases"]
        if row["stage"] not in stages:
            stages.append(row["stage"])
        result.row(
            row["stage"],
            row["metric"],
            row["severity"],
            f"{row['passed']}/{row['cases']}",
            _shown(row["mean_value"]),
            style="red" if failing and row["severity"] == "hard" else "yellow" if failing else "",
        )
    result.line("; ".join(gate.reasons) or "every gate cleared")
    for stage in stages:
        result.line(f"Full report ({stage}): {_report_file(stage, run.run_id)}")
    return result


def _case(store: DuckDbEvalStore, case_id: str, run_id: str | None) -> CommandResult:
    run = store.get_run(run_id) if run_id else store.latest_run()
    if run is None:
        raise CommandError("No such eval run — /eval runs lists them.")
    found = [item for item in store.transcripts(run.run_id) if item.case_id == case_id]
    if not found:
        raise CommandError(f"No case '{case_id}' in {run.run_id}")
    transcript = found[0].transcript
    result = CommandResult(title=f"{case_id} in {run.run_id}")
    for reply in transcript.replies:
        result.line(f"agent: {reply}")
    result.headers = ["metric", "passed", "why"]
    for row in store.query(
        "SELECT metric, passed, rationale FROM v_scores WHERE run_id = ? AND case_id = ? "
        "ORDER BY passed, metric",
        (run.run_id, case_id),
    ):
        result.row(
            row["metric"],
            "pass" if row["passed"] else "FAIL",
            _shown(row["rationale"]),
            style="" if row["passed"] else "red",
        )
    return result


def _report_file(stage: str, run_id: str) -> str:
    """Where `make eval` wrote one section's report of this run (each stage writes its own)."""
    from agent_platform.evals.core.reporting.writer import report_path

    return str(report_path(stage, run_id))


def _shown(value: Any) -> str:
    if value is None:
        return "—"
    return f"{value:.1%}" if isinstance(value, float) else str(value)
