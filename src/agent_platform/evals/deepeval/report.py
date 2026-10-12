"""DeepEval's report: `deepeval-<run_id>.md`, every case with its scores, and an overview."""

from pathlib import Path
from typing import Any

from agent_platform.config import get_settings
from agent_platform.evals.core.pipeline.gate import evaluate_gate
from agent_platform.evals.core.reporting import Overview, ReportSections, fill, write_report
from agent_platform.evals.core.store import DuckDbEvalStore, RunRecord
from agent_platform.judge import JudgeCalibrator

FRAMEWORK = STAGE = "deepeval"
_GENERATE = "generate"
_TEMPLATE = Path(__file__).resolve().parent / "template.md"
_SUPPORTED, _UNSUPPORTED = "supported", "unsupported"


class DeepEvalReport:
    """Writes DeepEval's view of a golden run and builds its terminal overview."""

    def __init__(self, store: DuckDbEvalStore) -> None:
        self._store = store
        self._sections = ReportSections(store)
        self._settings = get_settings().evals

    def write(self, run: RunRecord) -> Path:
        gate = evaluate_gate(self._store, run.run_id, STAGE)
        sections = self._sections
        body = fill(
            _TEMPLATE,
            run_id=run.run_id,
            verdict=gate.verdict,
            reasons="; ".join(gate.reasons) or "every gate cleared",
            judge_model=run.judge_model,
            soft_threshold=f"{gate.soft_threshold:.0%}",
            case_count=run.case_count,
            metric_names=", ".join(f"`{name}`" for name in self._metrics(run.run_id)),
            run_facts=sections.run_facts(run),
            changes=sections.changes_since_previous(run),
            deepeval=sections.metric_table(run.run_id, STAGE),
            deepeval_matrix=sections.category_matrix(run.run_id, STAGE),
            cases=sections.case_results(run.run_id, STAGE, scores=True),
            trials=sections.trials(run.run_id, STAGE),
            agreement=self._agreement(run.run_id),
            failures=sections.failing_cases(run.run_id, (STAGE,)),
            cost=sections.cost_table(run.run_id, (_GENERATE, STAGE)),
        )
        return write_report(FRAMEWORK, run.run_id, body)

    def overview(self, run: RunRecord, report: Path) -> Overview:
        gate = evaluate_gate(self._store, run.run_id, STAGE)
        sections = self._sections
        overview = Overview(f"DeepEval · {run.run_id}")
        overview.metrics("Judged quality (soft)", sections.metric_rows(run.run_id, STAGE))
        overview.failures(sections.failures(run.run_id, STAGE), report)
        facts = sections.facts_line(run, (_GENERATE, STAGE), judge=True)
        return overview.gate(gate, facts, report)

    def _metrics(self, run_id: str) -> list[str]:
        """The metrics this run was graded on (the core ones, or every one in a full run)."""
        rows = self._store.query(
            "SELECT DISTINCT metric FROM v_scores WHERE run_id = ? AND stage = ? ORDER BY metric",
            (run_id, STAGE),
        )
        return [row["metric"] for row in rows]

    def _agreement(self, run_id: str) -> str:
        """Cohen's κ between the judge's faithfulness verdicts and the runtime guard."""
        reporting = self._settings.reporting
        judged = {
            row["case_id"]: bool(row["passed"])
            for row in self._store.query(
                "SELECT case_id, passed FROM v_scores WHERE run_id = ? AND metric = ?",
                (run_id, reporting.guard_metric),
            )
        }
        pairs: list[tuple[str, str]] = []
        for item in self._store.transcripts(run_id):
            guard: Any = item.transcript.metadata.get(reporting.runtime_guard_key)
            if item.case_id in judged and guard is not None:
                pairs.append((_label(judged[item.case_id]), _label(float(guard) <= 0.0)))
        if not pairs:
            return "_No case carries both a faithfulness verdict and a runtime guard reading._"
        report = JudgeCalibrator().agreement([p[0] for p in pairs], [p[1] for p in pairs])
        disagreements = sum(1 for judge, guard in pairs if judge != guard)
        return (
            f"{report.sample_size} cases · observed agreement {report.observed_agreement:.1%} · "
            f"Cohen's κ {report.cohens_kappa:.2f} "
            f"({'acceptable' if report.acceptable else 'below the acceptable bar'}) · "
            f"{disagreements} disagreement(s). The guard runs in the request path with a local "
            "NLI model; the judge reads the whole conversation afterwards. Where they disagree, "
            "one of them is wrong — read those cases."
        )


def _label(supported: bool) -> str:
    return _SUPPORTED if supported else _UNSUPPORTED
