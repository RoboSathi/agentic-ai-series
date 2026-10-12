"""The oracle's report: `oracle-<run_id>.md`, every case with its verdict, and an overview."""

from pathlib import Path

from agent_platform.evals.core.pipeline.gate import evaluate_gate
from agent_platform.evals.core.reporting import Overview, ReportSections, fill, write_report
from agent_platform.evals.core.store import DuckDbEvalStore, RunRecord

FRAMEWORK = STAGE = "oracle"
_GENERATE = "generate"
_TEMPLATE = Path(__file__).resolve().parent / "template.md"


class OracleReport:
    """Writes the oracle's view of a golden run and builds its terminal overview."""

    def __init__(self, store: DuckDbEvalStore) -> None:
        self._store = store
        self._sections = ReportSections(store)

    def write(self, run: RunRecord) -> Path:
        gate = evaluate_gate(self._store, run.run_id, STAGE)
        sections = self._sections
        body = fill(
            _TEMPLATE,
            run_id=run.run_id,
            verdict=gate.verdict,
            reasons="; ".join(gate.reasons) or "every invariant held",
            soft_threshold=f"{gate.soft_threshold:.0%}",
            run_facts=sections.run_facts(run),
            changes=sections.changes_since_previous(run),
            rules=sections.metric_table(run.run_id, STAGE),
            matrix=sections.category_matrix(run.run_id, STAGE),
            cases=sections.case_results(run.run_id, STAGE),
            trials=sections.trials(run.run_id, STAGE),
            failures=sections.failing_cases(run.run_id, (STAGE,)),
            cost=sections.cost_table(run.run_id, (_GENERATE, STAGE)),
        )
        return write_report(FRAMEWORK, run.run_id, body)

    def overview(self, run: RunRecord, report: Path) -> Overview:
        gate = evaluate_gate(self._store, run.run_id, STAGE)
        sections = self._sections
        overview = Overview(f"Oracle · {run.run_id}")
        overview.metrics(
            "Business invariants (deterministic)", sections.metric_rows(run.run_id, STAGE)
        )
        overview.failures(sections.failures(run.run_id, STAGE), report)
        facts = sections.facts_line(run, (_GENERATE, STAGE), judge=False)
        return overview.gate(gate, facts, report)
