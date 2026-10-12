"""`python -m agent_platform.evals <command>` — what `make eval <agent>` runs.

The agent is found through `evals.plugin` in configuration, so this module
never names one. Every suite ends with a colour-coded overview in the
terminal and the path of its full Markdown report — all of them together,
once the progress bars are done — and then how long each section took.

    run [--section all|oracle|deepeval|promptfoo|intent] [--full]
                              the sections (default: oracle, DeepEval, then Promptfoo), over
                              the agent's core cases, attacks and metrics — or, with --full,
                              every one of them
    report [--run RUN_ID]     rebuild a run's reports
    runs                      recent runs

A failed gate is a result, not an error: the exit code is non-zero only
when a section could not run at all (every other section still runs).
While it runs, one progress bar per section shows the percentage done and
the time taken; the console shows warnings only (`evals.console_level` —
a 429 and the model it hit, a model or guard fallback), the log files
everything. Ctrl+C stops cleanly: the run is marked interrupted, the time so
far is shown, and the next `make eval` resumes from what finished.
"""

import argparse
import asyncio
import time
from collections.abc import Callable, Sequence

from agent_platform.config import get_settings
from agent_platform.evals.core.contracts import EvalPlugin
from agent_platform.evals.core.pipeline.orchestrator import (
    AGENT_SECTION,
    DEEPEVAL_SECTION,
    GOLDEN_SECTIONS,
    INTENT_SECTION,
    ORACLE_SECTION,
    PROMPTFOO_SECTION,
    EvalPipeline,
    NoIntentBakeoffError,
    NoRunError,
    SectionClock,
    StageOutcome,
)
from agent_platform.evals.core.plugin import resolve_eval_plugin
from agent_platform.evals.core.reporting import Overview
from agent_platform.evals.core.reporting.progress_bar import ProgressBar
from agent_platform.evals.core.store import DuckDbEvalStore
from agent_platform.logging import LoggerFactory, shared_console

_OK, _FAILED, _INTERRUPTED = 0, 1, 130
_SECONDS_PER_MINUTE, _SECONDS_PER_HOUR = 60, 3600
ALL, TOTAL = "all", "total"
SECTIONS = (ALL, ORACLE_SECTION, DEEPEVAL_SECTION, PROMPTFOO_SECTION, INTENT_SECTION)

console = shared_console()


def main(argv: Sequence[str] | None = None) -> int:
    """Parse, dispatch, and turn a missing run into a readable message."""
    args = _parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (NoRunError, NoIntentBakeoffError) as exc:
        console.print(f"[yellow]{exc}[/yellow]")
        return _FAILED


def _run(args: argparse.Namespace) -> int:
    """The chosen suites, each one run even when another raised; then every section's
    overview together, and the timing table."""
    started = time.perf_counter()
    settings = get_settings().evals
    LoggerFactory.set_console_level(settings.console_level)
    LoggerFactory.log_everything_to(settings.log_file)
    outcomes: list[StageOutcome] = []
    with ProgressBar() as progress:
        pipeline = EvalPipeline(_prepared_plugin(), progress=progress)
        try:
            errors = _run_sections(args.section, args.full, pipeline, outcomes)
        except KeyboardInterrupt:
            progress.stop()
            _show_all(outcomes)
            _print_interrupted(pipeline, args.invocation)
            _print_timings(pipeline.clock, time.perf_counter() - started)
            return _INTERRUPTED
    _show_all(outcomes)
    _print_timings(pipeline.clock, time.perf_counter() - started)
    return _FAILED if errors else _OK


def _print_interrupted(pipeline: EvalPipeline, invocation: str) -> None:
    """What stopped, what was kept, and the exact command that resumes it."""
    words = get_settings().evals.reporting.overview
    names = words.sections
    section = pipeline.clock.unfinished
    console.print(
        f"[yellow]{words.interrupted.format(section=names.get(section or '', section or ''), invocation=invocation or words.default_invocation)}[/yellow]"
    )
    marked = pipeline.store.mark_interrupted()
    if marked:
        console.print(
            f"[yellow]{words.marked_interrupted.format(stages=', '.join(marked))}[/yellow]"
        )


def _run_sections(
    section: str, full: bool, pipeline: EvalPipeline, outcomes: list[StageOutcome]
) -> int:
    """Run the chosen sections, keeping each overview as it finishes; return how many failed.

    Every section runs even when an earlier one raised.
    """
    errors = 0

    def shown(name: str, result: StageOutcome | Exception) -> None:
        nonlocal errors
        if isinstance(result, Exception):
            errors += 1
            _print_error(name, result)
        else:
            outcomes.append(result)

    golden = [name for name in GOLDEN_SECTIONS if section in (ALL, name)]
    work: list[tuple[str, Callable[[], object]]] = []
    if golden:
        work.append((AGENT_SECTION, lambda: asyncio.run(pipeline.golden(golden, full, shown))))
    if section in (ALL, PROMPTFOO_SECTION):
        work.append((PROMPTFOO_SECTION, lambda: shown(PROMPTFOO_SECTION, pipeline.security(full))))
    if section == INTENT_SECTION:
        work.append((INTENT_SECTION, lambda: shown(INTENT_SECTION, asyncio.run(pipeline.intent()))))
    for name, run in work:
        try:
            run()
        except Exception as exc:
            shown(name, exc)
    return errors


def _report(args: argparse.Namespace) -> int:
    for outcome in EvalPipeline(resolve_eval_plugin()).report(args.run):
        _show(outcome)
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


def _show_all(outcomes: list[StageOutcome]) -> None:
    """Every finished section's overview, in the order the sections ran."""
    for outcome in outcomes:
        _show(outcome)


def _show(outcome: StageOutcome) -> StageOutcome:
    outcome.overview.print(console)
    console.print()
    return outcome


def _print_error(section: str, exc: Exception) -> None:
    overview = get_settings().evals.reporting.overview
    name = overview.sections.get(section, section)
    message = overview.section_error.format(section=name, error=f"{type(exc).__name__}: {exc}")
    console.print(f"[red]{message}[/red]")


def _print_timings(clock: SectionClock, total: float) -> None:
    """How long each section took (the one Ctrl+C stopped, marked so), then the whole run."""
    words = get_settings().evals.reporting.overview
    names = words.sections
    rows = [
        [
            words.unfinished_section.format(section=names.get(section, section))
            if section == clock.unfinished
            else names.get(section, section),
            duration(seconds),
        ]
        for section, seconds in clock.elapsed.items()
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
    run.add_argument("--section", choices=SECTIONS, default=ALL)
    run.add_argument(
        "--full", action="store_true", help="every case, attack and metric, not just the core"
    )
    run.add_argument(
        "--invocation", default="", help="the command that started this run (shown to resume it)"
    )
    run.set_defaults(handler=_run)

    report = commands.add_parser("report", help="rebuild a run's reports and overviews")
    report.add_argument("--run", default=None, help="run id (default: latest golden run)")
    report.set_defaults(handler=_report)

    runs = commands.add_parser("runs", help="list recent runs")
    runs.add_argument("--limit", type=int, default=get_settings().evals.reporting.max_cases_listed)
    runs.set_defaults(handler=_runs)
    return parser
