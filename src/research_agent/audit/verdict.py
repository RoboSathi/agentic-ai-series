"""The audit's overall verdict: a weighted score, with must-pass criteria as a veto.

Each criterion's score (0..1) counts in proportion to its weight. A must-pass
criterion that fails caps the verdict at PARTIAL whatever the score, so a
safety-critical check (every claim cited, every claim supported) cannot be
averaged away by good coverage elsewhere. Thresholds are configuration
(`audit.pass_threshold`, `audit.partial_threshold`).
"""

from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, Field

from agent_platform.verification.acceptance import CheckVerdict
from research_agent.domain.rubric import Rubric

Outcome = Literal["PASS", "PARTIAL", "FAIL"]
DEFAULT_WEIGHT = 1.0


class AuditVerdict(BaseModel):
    """The weighted score, the criteria that vetoed, and what that makes of the report."""

    outcome: Outcome
    score: float
    vetoed_by: list[str] = Field(default_factory=list)


def weighted_score(verdicts: Iterable[CheckVerdict], rubric: Rubric) -> float:
    """Σ weight·score / Σ weight over the criteria that had something to check.

    A criterion with nothing to check (no claims to cite, no links to resolve) passes
    without earning credit: counting its vacuous 1.0 would let a report that established
    nothing score well. 1.0 when there are no criteria at all; 0.0 when none was checkable.
    """
    scored = list(verdicts)
    if not scored:
        return 1.0
    total = earned = 0.0
    for verdict in scored:
        if not verdict.applicable:
            continue
        item = rubric.item(verdict.item_id)
        weight = item.weight if item is not None else DEFAULT_WEIGHT
        total += weight
        earned += weight * min(1.0, max(0.0, verdict.score))
    return earned / total if total else 1.0


def decide(
    verdicts: list[CheckVerdict], rubric: Rubric, pass_threshold: float, partial_threshold: float
) -> AuditVerdict:
    score = weighted_score(verdicts, rubric)
    vetoed = [v.item_id for v in verdicts if v.must_pass and not v.passed]
    if score < partial_threshold:
        outcome: Outcome = "FAIL"
    elif vetoed or score < pass_threshold:
        outcome = "PARTIAL"
    else:
        outcome = "PASS"
    return AuditVerdict(outcome=outcome, score=score, vetoed_by=vetoed)
