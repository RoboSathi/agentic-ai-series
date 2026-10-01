"""Reflection after each wave: cheap checks first, a model only for the gaps.

1. Deterministic coverage checks over the ledger → the Gap list.
2. The archetype stop rule, plus budget, rounds and cancellation → stop?
3. Only if continuing (and reflection is not withheld): the reflector
   model proposes follow-up sub-questions for unmet items only. Its
   proposals are filtered in code — a follow-up that serves no gap, or
   only repeats a tried query, is dropped.
"""

import re
from collections.abc import Callable

from pydantic import BaseModel, Field

from agentic_commons.logging import instrument
from agentic_commons.verification.acceptance import AcceptanceCheckRegistry, CheckVerdict
from research_agent.audit.checks import COVERAGE_CHECKS
from research_agent.audit.context import CheckContext
from research_agent.domain.plan import SubQuestion
from research_agent.domain.scope import Archetype
from research_agent.prompts import reflector as prompt
from research_agent.prompts.base import RolePrompt
from research_agent.research.stop_rules import StopDecision, StopInput, StopRule
from research_agent.structured import StructuredCaller


class Reflection(BaseModel):
    """What one reflection step concluded."""

    verdicts: list[CheckVerdict] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    stop: bool = False
    reason: str = ""
    follow_ups: list[SubQuestion] = Field(default_factory=list)
    entities: int = 0


@instrument("research_agent")
class GapAnalyzer:
    """Coverage verdicts over the ledger, and the gaps among them."""

    def __init__(self, registry: AcceptanceCheckRegistry[CheckContext]) -> None:
        self._registry = registry

    async def verdicts(self, ctx: CheckContext) -> list[CheckVerdict]:
        specs = [spec for spec in ctx.contract.rubric.specs() if spec.name in COVERAGE_CHECKS]
        return await self._registry.evaluate(specs, ctx, kinds=["deterministic"])

    @staticmethod
    def entities(ctx: CheckContext, pattern: re.Pattern[str]) -> int:
        return len({name for entry in ctx.ledger.entries for name in pattern.findall(entry.claim)})


@instrument("research_agent")
class Reflector:
    """Decides continue/stop and proposes follow-ups for unmet items."""

    def __init__(
        self,
        caller: StructuredCaller | None,
        rule_for: Callable[[Archetype], StopRule],
        max_follow_ups: int,
    ) -> None:
        self._caller = caller
        self._rule_for = rule_for
        self._max = max_follow_ups

    def decide(
        self, archetype: Archetype, verdicts: list[CheckVerdict], before: int, after: int
    ) -> StopDecision:
        """The archetype's stop rule (Strategy), applied to this wave's verdicts."""
        observed = StopInput(verdicts=verdicts, entities_before=before, entities_after=after)
        return self._rule_for(archetype).decide(observed)

    async def follow_ups(
        self, ctx: CheckContext, verdicts: list[CheckVerdict], tried: list[str], round_number: int
    ) -> list[SubQuestion]:
        """Model-proposed follow-ups, filtered to unmet items and untried queries."""
        gaps = [v for v in verdicts if not v.passed]
        if self._caller is None or not gaps or self._max == 0:
            return []
        answer = await self._caller.call(self._prompt(ctx, gaps, tried), prompt.ReflectorAnswer)
        return _accepted(answer, {v.item_id for v in gaps}, set(tried), round_number, self._max)

    def _prompt(self, ctx: CheckContext, gaps: list[CheckVerdict], tried: list[str]) -> RolePrompt:
        payload_gaps = [
            {
                "item_id": v.item_id,
                "check": v.name,
                "missing": v.evidence,
                "text": _text(ctx, v.item_id),
            }
            for v in gaps
        ]
        claims = [entry.claim for entry in ctx.ledger.entries]
        return prompt.build(ctx.contract, payload_gaps, tried, claims, self._max)


def _text(ctx: CheckContext, item_id: str) -> str:
    item = ctx.contract.rubric.item(item_id)
    return item.text if item is not None else ""


def _accepted(
    answer: prompt.ReflectorAnswer,
    gap_ids: set[str],
    tried: set[str],
    round_number: int,
    limit: int,
) -> list[SubQuestion]:
    accepted: list[SubQuestion] = []
    for index, follow in enumerate(answer.follow_ups, start=1):
        serves = tuple(item for item in follow.rubric_items if item in gap_ids)
        fresh = tuple(
            query
            for query in follow.queries
            if query.casefold() not in {t.casefold() for t in tried}
        )
        if serves and fresh and len(accepted) < limit:
            accepted.append(
                SubQuestion(
                    node_id=f"F{round_number}-{index}",
                    question=follow.question,
                    rubric_items=serves,
                    queries=fresh,
                    subject=follow.subject,
                )
            )
    return accepted
