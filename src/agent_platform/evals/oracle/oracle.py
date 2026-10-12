"""The deterministic oracle: an agent's hard rules, run as a grader.

Zero model calls. The oracle is the only grader that reads ground truth —
the state a case left behind, checked against facts the agent did not
produce — so it is where "did money move that should not have?" is
answered. A judge reads only the conversation, and can be talked into
anything a conversation can say.

The platform owns the mechanics (run every rule, skip the ones with
nothing to say, score 1 or 0, version the rule set); an agent owns the
rules and the context they read. A rule grades the *outcome* — what the
agent left in the world — not the path it took to get there.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

from agent_platform.evals.core.contracts import (
    GoldenCase,
    GraderKind,
    Score,
    Severity,
    Transcript,
    canonical_hash,
)

ContextT = TypeVar("ContextT", contravariant=True)
OracleContextT = TypeVar("OracleContextT")

_PASS, _FAIL = 1.0, 0.0
_VERSION_CHARS = 16


@dataclass(frozen=True)
class RuleVerdict:
    """One rule's finding on one case."""

    passed: bool
    rationale: str


class Rule(Protocol[ContextT]):
    """One invariant. `None` means it has nothing to say about this case."""

    metric: str
    severity: Severity

    def check(self, ctx: ContextT) -> RuleVerdict | None: ...


class InvariantOracle(Generic[OracleContextT]):
    """Grades a transcript against every rule that applies to its case."""

    kind: GraderKind = "oracle"

    def __init__(
        self,
        name: str,
        rules: Sequence[Rule[OracleContextT]],
        context: Callable[[GoldenCase, Transcript], OracleContextT],
        version_inputs: Any = None,
    ) -> None:
        """`version_inputs`: whatever else changes a verdict (fixture facts, patterns)."""
        self.name = name
        self._rules = tuple(rules)
        self._context = context
        # Any change to the rules or their inputs regrades.
        self.version = canonical_hash(
            [[type(rule).__name__ for rule in self._rules], version_inputs]
        )[:_VERSION_CHARS]

    @property
    def metrics(self) -> list[str]:
        return [rule.metric for rule in self._rules]

    async def grade(self, case: GoldenCase, transcript: Transcript) -> list[Score]:
        ctx = self._context(case, transcript)
        return [
            self._score(case, rule, verdict)
            for rule in self._rules
            if (verdict := rule.check(ctx)) is not None
        ]

    def check(self, metric: str, case: GoldenCase, transcript: Transcript) -> RuleVerdict | None:
        """One rule by name — how another suite (Promptfoo) reuses a single invariant."""
        rule = next((rule for rule in self._rules if rule.metric == metric), None)
        if rule is None:
            raise KeyError(f"No oracle rule '{metric}' (rules: {', '.join(self.metrics)})")
        return rule.check(self._context(case, transcript))

    def _score(self, case: GoldenCase, rule: Rule[OracleContextT], verdict: RuleVerdict) -> Score:
        return Score(
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
