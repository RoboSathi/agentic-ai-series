"""The red-team suite's report: `promptfoo-<run_id>.md` and an overview."""

from collections.abc import Callable
from pathlib import Path

from agent_platform.config import get_settings
from agent_platform.evals.core.contracts import GoldenCase
from agent_platform.evals.core.pipeline.gate import evaluate_gate
from agent_platform.evals.core.reporting import Overview, ReportSections, fill, write_report
from agent_platform.evals.core.reporting.markdown import rate, table
from agent_platform.evals.core.reporting.overview import BREACHED_STATE, DEFENDED_STATE
from agent_platform.evals.core.store import DuckDbEvalStore, RunRecord
from agent_platform.evals.promptfoo.runner import severity_of, strategy_of

FRAMEWORK = "promptfoo"
_TEMPLATE = Path(__file__).resolve().parent / "template.md"
_STAGE = "promptfoo"
_UNDECLARED = "—"


class PromptfooReport:
    """Writes the red-team run's report and prints its overview."""

    def __init__(self, store: DuckDbEvalStore) -> None:
        self._store = store
        self._sections = ReportSections(store)

    def write(self, run: RunRecord) -> Path:
        gate = evaluate_gate(self._store, run.run_id)
        sections = self._sections
        body = fill(
            _TEMPLATE,
            run_id=run.run_id,
            verdict=gate.verdict,
            reasons="; ".join(gate.reasons) or "no attack got through",
            package=get_settings().evals.promptfoo.package,
            run_facts=sections.run_facts(run),
            plugins=self._grouped(run.run_id, "plugin", lambda case: case.category),
            severities=self._grouped(run.run_id, "severity", severity_of),
            strategies=self._grouped(run.run_id, "strategy", strategy_of),
            assertions=sections.metric_table(run.run_id, _STAGE),
            failures=sections.failing_cases(run.run_id, (_STAGE,)),
            cost=sections.cost_table(run.run_id),
        )
        return write_report(FRAMEWORK, run.run_id, body)

    def overview(self, run: RunRecord, report: Path) -> Overview:
        overview = Overview(f"Red team · {run.run_id}")
        rows = [
            [group, str(attacks), str(attacks - breached), str(breached), overview.word(state)]
            for group, attacks, breached in self._counts(run.run_id, lambda case: case.category)
            for state in [BREACHED_STATE if breached else DEFENDED_STATE]
        ]
        overview.table(
            "Promptfoo (by red-team plugin)",
            ["plugin", "attacks", "defended", "breached", "state"],
            rows,
        )
        overview.failures(self._sections.failures(run.run_id, _STAGE), report)
        facts = f"{run.case_count} attacks · agent {run.model} · judge {run.judge_model}"
        return overview.gate(evaluate_gate(self._store, run.run_id), facts, report)

    def _grouped(self, run_id: str, label: str, key: Callable[[GoldenCase], str]) -> str:
        z = self._sections.z
        return table(
            [label, "attacks", "got through", "defended (95% CI)"],
            [
                [group, attacks, breached, rate(attacks - breached, attacks, z)]
                for group, attacks, breached in self._counts(run_id, key)
            ],
        )

    def _counts(self, run_id: str, key: Callable[[GoldenCase], str]) -> list[tuple[str, int, int]]:
        """(group, attacks, attacks that got through), per group."""
        breached = {
            row["case_id"]
            for row in self._store.query(
                "SELECT DISTINCT case_id FROM v_scores WHERE run_id = ? AND stage = ? "
                "AND NOT passed",
                (run_id, _STAGE),
            )
        }
        groups: dict[str, list[str]] = {}
        for case_id, case in self._store.cases_for_run(run_id).items():
            groups.setdefault(key(case) or _UNDECLARED, []).append(case_id)
        return [
            (group, len(ids), sum(1 for case_id in ids if case_id in breached))
            for group, ids in sorted(groups.items())
        ]
