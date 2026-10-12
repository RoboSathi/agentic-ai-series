"""The golden suite's report: oracle + DeepEval, as `deepeval-<run_id>.md` and an overview."""

from pathlib import Path
from typing import Any

from agent_platform.config import get_settings
from agent_platform.evals.core.pipeline.gate import evaluate_gate
from agent_platform.evals.core.pipeline.trials import base_case_id
from agent_platform.evals.core.reporting import Overview, ReportSections, fill, write_report
from agent_platform.evals.core.reporting.markdown import table
from agent_platform.evals.core.store import DuckDbEvalStore, RunRecord
from agent_platform.judge import JudgeCalibrator

FRAMEWORK = "deepeval"
_TEMPLATE = Path(__file__).resolve().parent / "template.md"
_ORACLE, _DEEPEVAL = "oracle", "deepeval"
_SUPPORTED, _UNSUPPORTED = "supported", "unsupported"


class DeepEvalReport:
    """Writes the golden run's report and prints its overview."""

    def __init__(self, store: DuckDbEvalStore) -> None:
        self._store = store
        self._sections = ReportSections(store)
        self._settings = get_settings().evals

    def write(self, run: RunRecord) -> Path:
        gate = evaluate_gate(self._store, run.run_id)
        sections = self._sections
        body = fill(
            _TEMPLATE,
            run_id=run.run_id,
            verdict=gate.verdict,
            reasons="; ".join(gate.reasons) or "every gate cleared",
            judge_model=run.judge_model,
            soft_threshold=f"{gate.soft_threshold:.0%}",
            case_count=run.case_count,
            run_facts=sections.run_facts(run),
            changes=sections.changes_since_previous(run),
            oracle=sections.metric_table(run.run_id, _ORACLE),
            oracle_matrix=sections.category_matrix(run.run_id, _ORACLE),
            deepeval=sections.metric_table(run.run_id, _DEEPEVAL),
            deepeval_matrix=sections.category_matrix(run.run_id, _DEEPEVAL),
            trials=self._trials(run.run_id),
            agreement=self._agreement(run.run_id),
            failures=sections.failing_cases(run.run_id, (_ORACLE, _DEEPEVAL)),
            cost=sections.cost_table(run.run_id),
        )
        return write_report(FRAMEWORK, run.run_id, body)

    def overview(self, run: RunRecord, report: Path) -> Overview:
        gate = evaluate_gate(self._store, run.run_id)
        sections = self._sections
        overview = Overview(f"Golden set · {run.run_id}")
        overview.metrics("Oracle (deterministic, hard)", sections.metric_rows(run.run_id, _ORACLE))
        quality = sections.metric_rows(run.run_id, _DEEPEVAL)
        if quality:
            overview.metrics("DeepEval (judged, soft)", quality)
        failures = [
            *sections.failures(run.run_id, _ORACLE),
            *sections.failures(run.run_id, _DEEPEVAL),
        ]
        overview.failures(failures, report)
        return overview.gate(gate, self._facts(run), report)

    def _facts(self, run: RunRecord) -> str:
        spend = self._store.query(
            "SELECT sum(provider_requests) AS requests, sum(http_429) AS rejected, "
            "sum(seconds) AS seconds FROM v_stage_cost WHERE run_id = ? "
            "AND batch_id IN (SELECT batch_id FROM v_latest_batches)",
            (run.run_id,),
        )[0]
        return (
            f"{run.case_count} cases · agent {run.model} · judge {run.judge_model} · "
            f"{spend['requests'] or 0} provider requests · {spend['rejected'] or 0} × 429 · "
            f"{spend['seconds'] or 0}s"
        )

    def _trials(self, run_id: str) -> str:
        """pass@k and pass^k per base case, when cases ran more than once."""
        trials = self._settings.trials
        if trials <= 1:
            return f"_Each case ran once (`evals.trials: {trials}`); pass@k and pass^k need k > 1._"
        rows = self._store.query(
            "SELECT case_id, bool_and(passed) AS clean FROM v_scores WHERE run_id = ? "
            "AND severity = 'hard' GROUP BY case_id",
            (run_id,),
        )
        by_case: dict[str, list[bool]] = {}
        for row in rows:
            by_case.setdefault(base_case_id(row["case_id"]), []).append(bool(row["clean"]))
        any_pass = sum(any(results) for results in by_case.values())
        all_pass = sum(all(results) for results in by_case.values())
        total = len(by_case)
        return table(
            ["k", "cases", "pass@k (some trial clean)", "pass^k (every trial clean)"],
            [[trials, total, f"{any_pass}/{total}", f"{all_pass}/{total}"]],
        )

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
