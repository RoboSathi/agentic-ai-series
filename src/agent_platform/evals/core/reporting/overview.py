"""The overview `make eval` prints: a few colour-coded box tables per stage.

The Markdown report is the record; this is the glance. Both are built from
the same store views, and the tables go through the console's own
`CommandView`, so they look — and are coloured — exactly as `/review` and
`make show-memory` are: outcome words in their state colours
(`tui.command_output.state_styles`), everything else plain.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rich.console import Console

from agent_platform.config import get_settings
from agent_platform.evals.core.pipeline.gate import FAIL, PASS, GateResult
from agent_platform.interface.commands.types import CommandResult, CommandTable
from agent_platform.interface.presentation.terminal import print_result

PASS_STATE, FAIL_STATE, WARN_STATE = "pass", "fail", "warn"
DEFENDED_STATE, BREACHED_STATE, INCOMPLETE_STATE = "defended", "breached", "incomplete"
_HARD = "hard"
_PERCENT = 100


class Overview:
    """Builds and prints one stage's overview."""

    def __init__(self, title: str) -> None:
        settings = get_settings().evals.reporting.overview
        self._states = settings.states
        self._max_failures = settings.max_failures
        self._reason_chars = settings.reason_chars
        self._soft_bar = get_settings().evals.gate.soft_pass_rate
        self._result = CommandResult(title=title)

    def word(self, state: str) -> str:
        """An outcome's word, as configured (and so as coloured)."""
        return self._states[state]

    def metrics(self, title: str, rows: Sequence[dict[str, Any]]) -> "Overview":
        """One row per metric: kind, passed/total, rate, threshold, state."""
        table = CommandTable(
            title=title, headers=["metric", "kind", "passed", "rate", "threshold", "state"]
        )
        for row in rows:
            passed, total = int(row["passed"]), int(row["cases"])
            table.rows.append(
                [
                    str(row["metric"]),
                    str(row["severity"]),
                    f"{passed}/{total}",
                    _percent(passed, total),
                    _threshold(row.get("threshold")),
                    self.word(self._metric_state(row["severity"], passed, total)),
                ]
            )
        self._result.table(table)
        return self

    def table(self, title: str, headers: list[str], rows: list[list[str]]) -> "Overview":
        self._result.table(CommandTable(title=title, headers=headers, rows=rows))
        return self

    def failures(self, rows: Sequence[dict[str, Any]], report: Path) -> "Overview":
        """The first few failures, one line each; the rest live in the report."""
        if not rows:
            return self
        shown = [
            [
                str(row["case_id"]),
                str(row["category"]),
                str(row["metric"]),
                _clip(str(row.get("rationale") or ""), self._reason_chars),
            ]
            for row in rows[: self._max_failures]
        ]
        self.table("Failures", ["case", "category", "metric", "why"], shown)
        hidden = len(rows) - len(shown)
        if hidden > 0:
            self._result.line(f"+{hidden} more → {report}")
        return self

    def gate(self, gate: GateResult, facts: str, report: Path) -> "Overview":
        """The verdict line, the run's facts, and where the full report is."""
        state = {PASS: PASS_STATE, FAIL: FAIL_STATE}.get(gate.verdict, INCOMPLETE_STATE)
        reasons = "; ".join(gate.reasons) or "every gate cleared"
        self.table("Gate", ["verdict", "why"], [[self.word(state), reasons]])
        self._result.line(facts)
        self._result.line(f"Report: {report}")
        return self

    def print(self, console: Console | None = None) -> None:
        print_result(self._result, console)

    @property
    def result(self) -> CommandResult:
        return self._result

    def _metric_state(self, severity: str, passed: int, total: int) -> str:
        if total and passed == total:
            return PASS_STATE
        if severity == _HARD:
            return FAIL_STATE
        return PASS_STATE if total and passed / total >= self._soft_bar else WARN_STATE


def _percent(passed: int, total: int) -> str:
    return f"{passed / total * _PERCENT:.0f}%" if total else "—"


def _threshold(value: Any) -> str:
    return f"{float(value):g}" if value is not None else "—"


def _clip(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"
