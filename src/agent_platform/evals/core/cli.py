"""`python -m agent_platform.evals <command>` — what `make eval <agent>` runs.

The agent is found through `evals.plugin` in configuration, so this module
never names one. Every suite ends with a colour-coded overview in the
terminal and the path of its full Markdown report; the run ends with how
long each section took.

    run [--stage all|deepeval|promptfoo|intent]   the suites (default: golden, then red team)
    report [--run RUN_ID]                         rebuild a run's report
    runs                                          recent runs

A failed gate is a result, not an error: the exit code is non-zero only
when a section could not run at all (every other section still runs).
"""

import argparse
import asyncio
import time
from collections.abc import Callable, Sequence

from rich.console import Console

from agent_platform.config import get_settings
from agent_platform.evals.core.contracts import EvalPlugin
from agent_platform.evals.core.pipeline.orchestrator import (
    EvalPipeline,
    NoIntentBakeoffError,
    NoRunError,
    StageOutcome,
)
from agent_platform.evals.core.plugin import resolve_eval_plugin
from agent_platform.evals.core.reporting import Overview
from agent_platform.evals.core.store import DuckDbEvalStore

_OK, _FAILED = 0, 1
_SECONDS_PER_MINUTE, _SECONDS_PER_HOUR = 60, 3600
ALL, DEEPEVAL, PROMPTFOO, INTENT, TOTAL = "all", "deepeval", "promptfoo", "intent", "total"
STAGES = (ALL, DEEPEVAL, PROMPTFOO, INTENT)

console = Console(highlight=False)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse, dispatch, and turn a missing run into a readable message."""
    args = _parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (NoRunError, NoIntentBakeoffError) as exc:
        console.print(f"[yellow]{exc}[/yellow]")
        return _FAILED


def _run(args: argparse.Namespace) -> int:
    """The chosen suites, each one run even when another raised; then the timing table."""
    started = time.perf_counter()
    pipeline = EvalPipeline(_prepared_plugin())
    suites: list[tuple[str, Callable[[], StageOutcome]]] = []
    if args.stage in (ALL, DEEPEVAL):
        suites.append((DEEPEVAL, lambda: asyncio.run(pipeline.golden())))
    if args.stage in (ALL, PROMPTFOO):
        suites.append((PROMPTFOO, lambda: pipeline.security()))
    if args.stage == INTENT:
        suites.append((INTENT, lambda: asyncio.run(pipeline.intent())))
    outcomes, errors = [], 0
    for section, run in suites:
        try:
            outcomes.append(_show(run()))
        except Exception as exc:
            errors += 1
            _print_error(section, exc)
    _print_timings(outcomes, time.perf_counter() - started)
    return _FAILED if errors else _OK


def _report(args: argparse.Namespace) -> int:
    _show(EvalPipeline(resolve_eval_plugin()).report(args.run))
    return _OK


def _runs(args: argparse.Namespace) -> int:
    rows = DuckDbEvalStore().query(
        "SELECT run_id, suite, status, case_count, hard_failed_cases, soft_pass_rate "
        "FROM v_run_summary ORDER BY started_at DESC LIMIT ?",
        (args.limit,),
    )
    overview = Overview("Eval runs")
    overview.table(
        "Most recent first",
        ["run", "suite", "status", "cases", "hard-failed cases", "soft pass rate"],
        [["—" if value is None else str(value) for value in row.values()] for row in rows],
    )
    overview.print(console)
    return _OK


def _show(outcome: StageOutcome) -> StageOutcome:
    outcome.overview.print(console)
    console.print()
    return outcome


def _print_error(section: str, exc: Exception) -> None:
    overview = get_settings().evals.reporting.overview
    name = overview.sections.get(section, section)
    message = overview.section_error.format(section=name, error=f"{type(exc).__name__}: {exc}")
    console.print(f"[red]{message}[/red]")


def _print_timings(outcomes: list[StageOutcome], total: float) -> None:
    """How long each section took, then the whole run."""
    names = get_settings().evals.reporting.overview.sections
    rows = [
        [names.get(section, section), duration(seconds)]
        for outcome in outcomes
        for section, seconds in outcome.timings.items()
    ]
    overview = Overview("Time taken")
    overview.table("", ["section", "time"], [*rows, [names[TOTAL], duration(total)]])
    overview.print(console)


def duration(seconds: float) -> str:
    """`8m 26s`, `42s`, `1h 03m` — readable at a glance."""
    whole = round(seconds)
    hours, rest = divmod(whole, _SECONDS_PER_HOUR)
    minutes, secs = divmod(rest, _SECONDS_PER_MINUTE)
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m {secs:02d}s" if minutes else f"{secs}s"


def _prepared_plugin() -> EvalPlugin:
    plugin = resolve_eval_plugin()
    plugin.prepare()
    return plugin


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m agent_platform.evals")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run the agent's eval suites")
    run.add_argument("--stage", choices=STAGES, default=ALL)
    run.set_defaults(handler=_run)

    report = commands.add_parser("report", help="rebuild a run's report and overview")
    report.add_argument("--run", default=None, help="run id (default: latest golden run)")
    report.set_defaults(handler=_report)

    runs = commands.add_parser("runs", help="list recent runs")
    runs.add_argument("--limit", type=int, default=get_settings().evals.reporting.max_cases_listed)
    runs.set_defaults(handler=_runs)
    return parser
