"""`python -m agent_platform.evals <command>` — what the `make eval-*` targets run.

The agent is found through `evals.plugin` in configuration, so this
module never names one. Every stage prints where its report landed.
"""

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console
from rich.table import Table

from agent_platform.config import get_settings
from agent_platform.evals.contracts import EvalPlugin, GoldenCase, StageName
from agent_platform.evals.pipeline.gate import FAIL, GateResult, evaluate_gate
from agent_platform.evals.pipeline.orchestrator import (
    EvalPipeline,
    NoRunError,
    native_security_suite,
)
from agent_platform.evals.plugin import resolve_eval_plugin
from agent_platform.evals.store import DuckDbEvalStore

_OK, _FAILED = 0, 1
_GRADING_STAGES: tuple[StageName, ...] = ("oracle", "judge", "deepeval")
_VERDICT_STYLE = {"PASS": "green", "FAIL": "red", "INCOMPLETE": "yellow"}

console = Console()


def main(argv: Sequence[str] | None = None) -> int:
    """Parse, dispatch, and turn a missing run into a readable message."""
    args = _parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except NoRunError as exc:
        console.print(f"[yellow]{exc}[/yellow]")
        return _FAILED


# -- commands ------------------------------------------------------------------


def _generate(args: argparse.Namespace) -> int:
    plugin = _prepared_plugin()
    cases = _select(plugin.load_cases(args.suite), args)
    pipeline = EvalPipeline(plugin)
    run = asyncio.run(pipeline.generate(args.suite, cases, fresh=args.fresh))
    _print_run(pipeline, run.run_id)
    return _OK


def _grade(args: argparse.Namespace) -> int:
    pipeline = EvalPipeline(resolve_eval_plugin())
    gate = asyncio.run(pipeline.grade(args.stage, args.run, fresh=args.fresh))
    _print_run(pipeline, pipeline.run_or_latest(args.run).run_id)
    return _FAILED if args.stage == "oracle" and pipeline.should_stop(gate) else _OK


def _security(args: argparse.Namespace) -> int:
    plugin = _prepared_plugin()
    suite = native_security_suite(plugin)
    if suite is not None:
        return _native_security(plugin, suite, args)
    from agent_platform.evals.adapters.promptfoo import PromptfooSuite

    pipeline = EvalPipeline(plugin)
    run = PromptfooSuite(plugin, pipeline.store).run(fresh=args.fresh)
    EvalPipeline(plugin, pipeline.store).report(run.run_id)
    _print_run(pipeline, run.run_id)
    gate = evaluate_gate(pipeline.store, run.run_id, run.suite)
    return _FAILED if gate.verdict == FAIL else _OK


def _native_security(plugin: EvalPlugin, suite: str, args: argparse.Namespace) -> int:
    """A plugin's own security suite: generate its cases, grade them with its oracle."""
    pipeline = EvalPipeline(plugin)
    run = asyncio.run(pipeline.generate(suite, plugin.load_cases(suite), fresh=args.fresh))
    gate = asyncio.run(pipeline.grade("oracle", run.run_id, fresh=args.fresh))
    _print_run(pipeline, run.run_id)
    return _FAILED if gate.verdict == FAIL else _OK


def _run_all(args: argparse.Namespace) -> int:
    """The whole pipeline, in order, stopping after the oracle on a broken invariant."""
    suite = get_settings().evals.suites[0]
    args.suite, args.run = suite, None
    if _generate(args) != _OK:
        return _FAILED
    pipeline = EvalPipeline(resolve_eval_plugin())
    run_id = pipeline.run_or_latest(None, suite).run_id
    gate = asyncio.run(pipeline.grade("oracle", run_id, fresh=args.fresh))
    if pipeline.should_stop(gate):
        console.print(
            "[red]A hard business invariant failed — stopping before the paid stages.[/red] "
            "Set EVAL_CONTINUE_ON_INVARIANT_FAILURE=1 to run them anyway."
        )
        _print_run(pipeline, run_id)
        return _FAILED
    security = _security(args)
    for stage in ("judge", "deepeval"):
        asyncio.run(pipeline.grade(stage, run_id, fresh=args.fresh))
    _print_run(pipeline, run_id)
    final = evaluate_gate(pipeline.store, run_id, suite)
    return _FAILED if final.verdict == FAIL or security != _OK else _OK


def _report(args: argparse.Namespace) -> int:
    pipeline = EvalPipeline(resolve_eval_plugin())
    run_id = pipeline.run_or_latest(args.run).run_id
    for path in pipeline.report(run_id):
        console.print(f"  {path}")
    return _OK


def _diff(args: argparse.Namespace) -> int:
    pipeline = EvalPipeline(resolve_eval_plugin())
    console.print(f"  {pipeline.diff(args.base, args.head)}")
    return _OK


def _runs(args: argparse.Namespace) -> int:
    store = DuckDbEvalStore()
    rows = store.query(
        "SELECT * FROM v_run_summary ORDER BY started_at DESC LIMIT ?", (args.limit,)
    )
    _print_rows("Eval runs", rows)
    return _OK


def _sql(args: argparse.Namespace) -> int:
    """Run one query, or read queries from stdin until EOF."""
    store = DuckDbEvalStore()
    queries = [" ".join(args.query)] if args.query else _stdin_queries()
    for query in queries:
        _print_rows(query, store.query(query))
    return _OK


# -- helpers ---------------------------------------------------------------------


def _prepared_plugin() -> EvalPlugin:
    plugin = resolve_eval_plugin()
    plugin.prepare()
    return plugin


def _select(cases: list[GoldenCase], args: argparse.Namespace) -> list[GoldenCase]:
    selected = [
        case
        for case in cases
        if (not args.case or case.case_id in args.case)
        and (not args.tag or set(args.tag) & set(case.tags))
        and (not args.category or case.category in args.category)
    ]
    return selected[: args.limit] if args.limit else selected


def _print_run(pipeline: EvalPipeline, run_id: str) -> None:
    run = pipeline.run_or_latest(run_id)
    gate = evaluate_gate(pipeline.store, run.run_id, run.suite)
    _print_gate(run.run_id, gate)
    folder = get_settings().evals.reporting.resolved_output_dir / run.run_id
    for path in sorted(Path(folder).glob("*.md")):
        console.print(f"  report: {path}")


def _print_gate(run_id: str, gate: GateResult) -> None:
    style = _VERDICT_STYLE.get(gate.verdict, "white")
    reasons = "; ".join(gate.reasons) or "every gate cleared"
    console.print(f"[bold]{run_id}[/bold]: [{style}]{gate.verdict}[/{style}] — {reasons}")


def _print_rows(title: str, rows: list[dict[str, object]]) -> None:
    table = Table(title=title)
    for column in rows[0].keys() if rows else ["(no rows)"]:
        table.add_column(str(column), overflow="fold")
    for row in rows:
        table.add_row(*(str(value) for value in row.values()))
    console.print(table)


def _stdin_queries() -> list[str]:
    console.print(
        "[dim]One query per line; Ctrl-D to finish. Views: v_run_summary, "
        "v_metric_pass_rates, v_category_breakdown, v_failures, v_hard_failures, "
        "v_cost_latency, v_stage_cost, v_metric_trend.[/dim]"
    )
    return [line.strip() for line in sys.stdin if line.strip()]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m agent_platform.evals")
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate", help="run the agent over a suite")
    generate.add_argument("--suite", default=get_settings().evals.suites[0])
    _add_selection(generate)
    generate.set_defaults(handler=_generate)

    for stage in _GRADING_STAGES:
        grade = commands.add_parser(stage, help=f"score a run with the {stage} stage")
        grade.add_argument("--run", default=None, help="run id (default: latest)")
        grade.add_argument("--fresh", action="store_true", help="ignore cached scores")
        grade.set_defaults(handler=_grade, stage=stage)

    security = commands.add_parser("security", help="run the Promptfoo security suite")
    security.add_argument("--fresh", action="store_true")
    security.set_defaults(handler=_security)

    run = commands.add_parser("run", help="every stage, in pipeline order")
    _add_selection(run)
    run.set_defaults(handler=_run_all)

    report = commands.add_parser("report", help="rebuild a run's Markdown reports")
    report.add_argument("--run", default=None)
    report.set_defaults(handler=_report)

    diff = commands.add_parser("diff", help="compare two runs")
    diff.add_argument("base")
    diff.add_argument("head")
    diff.set_defaults(handler=_diff)

    runs = commands.add_parser("runs", help="list recent runs")
    runs.add_argument("--limit", type=int, default=get_settings().evals.reporting.max_cases_listed)
    runs.set_defaults(handler=_runs)

    sql = commands.add_parser("sql", help="query the eval store (views: v_*)")
    sql.add_argument("query", nargs="*")
    sql.set_defaults(handler=_sql)
    return parser


def _add_selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fresh", action="store_true", help="ignore cached transcripts/scores")
    parser.add_argument("--limit", type=int, default=0, help="only the first N cases")
    parser.add_argument("--case", action="append", default=[], help="only this case id")
    parser.add_argument("--tag", action="append", default=[], help="only cases with this tag")
    parser.add_argument("--category", action="append", default=[], help="only this category")
