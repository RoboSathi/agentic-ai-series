"""The intent bake-off's report: `intent-<run_id>.md` and an overview."""

from pathlib import Path

from agent_platform.config import get_settings
from agent_platform.evals.core.reporting import Overview, write_report
from agent_platform.evals.intent.scoring import BakeoffResult, render_report

FRAMEWORK = "intent"
_MS_FORMAT = "{:.0f}"
_RATE_FORMAT = "{:.3f}"


def write(result: BakeoffResult, run_id: str) -> Path:
    """The full comparison: summary, recall by intent, threshold sweep, every miss."""
    z = get_settings().evals.reporting.confidence_z
    return write_report(
        FRAMEWORK, run_id, render_report(result.title, result.scores, result.misses, z)
    )


def overview(result: BakeoffResult, run_id: str, report: Path) -> Overview:
    """One row per contender and mode."""
    rows = [
        [
            name,
            f"{score.correct}/{score.n}",
            _RATE_FORMAT.format(score.macro_f1),
            _RATE_FORMAT.format(score.coverage),
            _RATE_FORMAT.format(score.ece),
            _MS_FORMAT.format(score.p95_ms),
        ]
        for name, score in result.scores.items()
    ]
    overview = Overview(f"Intent bake-off · {run_id}")
    overview.table(
        result.title,
        ["contender · mode", "correct", "macro-F1", "coverage", "ECE", "p95 ms"],
        rows,
    )
    overview.result.line(f"Report: {report}")
    return overview
