"""The gate — does this run pass?

Two rules, deliberately different in kind:

- **hard** metrics (the oracle's business invariants, Promptfoo's security
  assertions) gate at 100%: one failure fails the run, because "refunded
  the wrong order in 1 of 50 cases" is not a 98% score, it is a defect;
- **soft** metrics (DeepEval quality) gate at a pass rate,
  `evals.gate.soft_pass_rate` (an agent may raise it in its overlay).
"""

from pydantic import BaseModel, Field

from agent_platform.config import get_settings
from agent_platform.evals.core.store import DuckDbEvalStore

PASS, FAIL, INCOMPLETE = "PASS", "FAIL", "INCOMPLETE"


class GateResult(BaseModel):
    """A run's verdict, and every reason behind it."""

    verdict: str
    hard_failed_cases: int = 0
    hard_scores: int = 0
    soft_pass_rate: float | None = None
    soft_threshold: float
    reasons: list[str] = Field(default_factory=list)


def evaluate_gate(store: DuckDbEvalStore, run_id: str) -> GateResult:
    """Apply the gate to the latest scores of every stage that ran."""
    threshold = get_settings().evals.gate.soft_pass_rate
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
