"""The gate — does this run pass?

Two rules, deliberately different in kind:

- **hard** metrics (business invariants, security assertions) gate at
  100%: one failure fails the run, because "refunded the wrong order in
  1 of 50 cases" is not a 98% score, it is a defect;
- **soft** metrics (judge rubric, DeepEval) gate at a pass rate — the
  suite's `regression_threshold` from the eval manifest when it declares
  one, else `evals.gate.soft_pass_rate`.
"""

from pydantic import BaseModel, Field

from agentic_commons.config import get_settings
from agentic_commons.evals.manifest import EvalManifest, load_manifest
from agentic_commons.evals.store import DuckDbEvalStore

PASS, FAIL, INCOMPLETE = "PASS", "FAIL", "INCOMPLETE"


class GateResult(BaseModel):
    """A run's verdict, and every reason behind it."""

    verdict: str
    hard_failed_cases: int = 0
    hard_scores: int = 0
    soft_pass_rate: float | None = None
    soft_threshold: float
    reasons: list[str] = Field(default_factory=list)


def soft_threshold(suite: str, manifest: EvalManifest | None = None) -> float:
    """The suite's declared regression threshold, else the configured default."""
    declared = [
        entry.regression_threshold
        for entry in (manifest or load_manifest()).entries
        if entry.suite == suite
    ]
    return max(declared) if declared else get_settings().evals.gate.soft_pass_rate


def evaluate_gate(store: DuckDbEvalStore, run_id: str, suite: str) -> GateResult:
    """Apply the gate to the latest scores of every stage that ran."""
    threshold = soft_threshold(suite)
    hard = store.query(
        "SELECT count(*) AS scores, count(DISTINCT CASE WHEN NOT passed THEN case_id END) "
        "AS failed FROM v_scores WHERE run_id = ? AND severity = 'hard'",
        (run_id,),
    )[0]
    soft = store.query(
        "SELECT avg(CASE WHEN passed THEN 1.0 ELSE 0.0 END) AS rate FROM v_scores "
        "WHERE run_id = ? AND severity = 'soft'",
        (run_id,),
    )[0]
    rate = soft["rate"]
    reasons: list[str] = []
    if not hard["scores"] and rate is None:
        return GateResult(verdict=INCOMPLETE, soft_threshold=threshold, reasons=["no scores yet"])
    if hard["failed"]:
        reasons.append(f"{hard['failed']} case(s) broke a hard invariant")
    if rate is not None and rate < threshold:
        reasons.append(f"soft pass rate {rate:.1%} is below {threshold:.0%}")
    return GateResult(
        verdict=FAIL if reasons else PASS,
        hard_failed_cases=int(hard["failed"] or 0),
        hard_scores=int(hard["scores"] or 0),
        soft_pass_rate=float(rate) if rate is not None else None,
        soft_threshold=threshold,
        reasons=reasons,
    )
