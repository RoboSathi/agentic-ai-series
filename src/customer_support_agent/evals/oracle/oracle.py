"""The deterministic oracle — the business invariants as a grader.

Zero model calls. Each invariant is exact, so each score is 1 or 0 and
its tier is `deterministic_rule`: these results count toward the NOLLM
ratio, and a hard failure here is a regression, not a matter of taste.
"""

import re

from agentic_commons.evals.contracts import (
    GoldenCase,
    GraderKind,
    Score,
    Transcript,
    canonical_hash,
)
from customer_support_agent.config.settings import get_agent_settings
from customer_support_agent.evals.dataset.schema import CsaCasePayload
from customer_support_agent.evals.fixture import FixtureFacts
from customer_support_agent.evals.oracle.invariants import INVARIANTS, Invariant, OracleContext

_PASS, _FAIL = 1.0, 0.0


class CsaOracle:
    """Grades a transcript against every applicable business invariant."""

    name = "csa_oracle"
    kind: GraderKind = "oracle"

    def __init__(self, facts: FixtureFacts, invariants: tuple[Invariant, ...] = INVARIANTS) -> None:
        settings = get_agent_settings()
        self._facts = facts
        self._invariants = invariants
        self._orders = re.compile(settings.evals.order_reference_pattern)
        self._money = re.compile(settings.evals.money_pattern)
        self._max_rounds = settings.max_tool_iterations
        # Any change to the rules, the fixture or the patterns regrades.
        self.version = canonical_hash(
            [
                [type(rule).__name__ for rule in invariants],
                facts.model_dump(mode="json"),
                settings.evals.order_reference_pattern,
                settings.evals.money_pattern,
                self._max_rounds,
            ]
        )[:16]

    async def grade(self, case: GoldenCase, transcript: Transcript) -> list[Score]:
        payload = CsaCasePayload.of(case)
        context = OracleContext(
            case=case,
            expected=payload.expected,
            transcript=transcript,
            facts=self._facts,
            customer=self._facts.user(payload.customer),
            order_pattern=self._orders,
            money_pattern=self._money,
            max_tool_rounds=self._max_rounds,
        )
        scores: list[Score] = []
        for rule in self._invariants:
            verdict = rule.check(context)
            if verdict is None:
                continue
            scores.append(
                Score(
                    case_id=case.case_id,
                    metric=rule.metric,
                    grader=self.name,
                    kind=self.kind,
                    severity=rule.severity,
                    value=_PASS if verdict.passed else _FAIL,
                    passed=verdict.passed,
                    threshold=_PASS,
                    tier="deterministic_rule",
                    rationale=verdict.rationale,
                )
            )
        return scores
