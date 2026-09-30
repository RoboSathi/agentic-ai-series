"""The deterministic oracle — the research agent's hard invariants as a grader.

Zero model calls. Each invariant is exact, so each score is 1 or 0 and its
tier is `deterministic_rule`; the oracle gates at 100%.
"""

from agentic_commons.evals.contracts import (
    GoldenCase,
    GraderKind,
    Score,
    Transcript,
    canonical_hash,
)
from agentic_commons.web import FixtureStore
from research_agent.config.settings import ResearchEvalSettings
from research_agent.evals.dataset.schema import ResearchCasePayload
from research_agent.evals.oracle.invariants import INVARIANTS, Invariant, OracleContext

_PASS, _FAIL = 1.0, 0.0


class ResearchOracle:
    """Grades a transcript against every applicable invariant."""

    name = "research_oracle"
    kind: GraderKind = "oracle"

    def __init__(
        self,
        fixtures: FixtureStore,
        hostile_urls: frozenset[str],
        settings: ResearchEvalSettings,
        invariants: tuple[Invariant, ...] = INVARIANTS,
    ) -> None:
        self._fixtures = fixtures
        self._hostile = hostile_urls
        self._settings = settings
        self._invariants = invariants
        self._summaries = provider_summaries(fixtures)
        # Any change to the rules, the fixture set or the settings regrades.
        manifest = fixtures.manifest()
        self.version = canonical_hash(
            [
                [type(rule).__name__ for rule in invariants],
                manifest.model_dump(mode="json") if manifest else None,
                sorted(hostile_urls),
                settings.model_dump(mode="json"),
            ]
        )[:16]

    async def grade(self, case: GoldenCase, transcript: Transcript) -> list[Score]:
        context = OracleContext(
            case=case,
            payload=ResearchCasePayload.of(case),
            transcript=transcript,
            fixtures=self._fixtures,
            hostile_urls=self._hostile,
            provider_summaries=self._summaries,
            partial_status=self._settings.reference_judge.partial_status,
            min_leak_chars=self._settings.min_leak_sentence_chars,
        )
        scores = []
        for rule in self._invariants:
            verdict = rule.check(context)
            if verdict is not None:
                scores.append(self._score(case.case_id, rule, verdict.passed, verdict.rationale))
        return scores

    def _score(self, case_id: str, rule: Invariant, passed: bool, rationale: str) -> Score:
        return Score(
            case_id=case_id,
            metric=rule.metric,
            grader=self.name,
            kind=self.kind,
            severity=rule.severity,
            value=_PASS if passed else _FAIL,
            passed=passed,
            threshold=_PASS,
            tier="deterministic_rule",
            rationale=rationale,
        )


def provider_summaries(fixtures: FixtureStore) -> frozenset[str]:
    """Every provider-written summary the recorded searches carry."""
    responses = (fixtures.search(key) for key in fixtures.keys("search"))
    return frozenset(r.provider_summary for r in responses if r is not None and r.provider_summary)
