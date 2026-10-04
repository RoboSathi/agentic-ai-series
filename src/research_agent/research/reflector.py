"""Reflection after each batch: code checks first, then the model's judgement.

1. Deterministic coverage checks over the ledger → the Gap list.
2. The archetype stop rule, plus budget, rounds and cancellation → stop?
3. Gaps left (and reflection not withheld): the reflector model proposes
   follow-up sub-questions for the unmet items only.
4. Every check passed and the stop rule says stop: the reflector model
   critiques whether the claims actually answer the question — the checks
   count citations and terms, not meaning — and may propose follow-ups.

A model never declares a run done while a check fails: it is asked to
critique only once the code says the criteria are met. Its proposals are
filtered in code — a follow-up that serves no criterion, or only repeats a
tried query, is dropped.
"""

import re
from collections.abc import Callable

from pydantic import BaseModel, Field

from agent_platform.logging import instrument
from agent_platform.observability.step_card import ModelCall
from agent_platform.verification.acceptance import AcceptanceCheckRegistry, CheckVerdict
from research_agent.audit.checks import COVERAGE_CHECKS
from research_agent.audit.context import CheckContext
from research_agent.domain.plan import SubQuestion
from research_agent.domain.scope import Archetype
from research_agent.llm.structured import StructuredCaller
from research_agent.prompts import reflector as prompt
from research_agent.prompts.base import RolePrompt
from research_agent.research.stop_rules import StopDecision, StopInput, StopRule

# Why a proposed follow-up was dropped (`research_agent.messages` words them for the card).
NO_GAP = "no_gap"
TRIED = "tried"
OVER_LIMIT = "over_limit"


class Rejected(BaseModel):
    """A follow-up the model proposed and the code dropped, and why."""

    question: str
    why: str


class FollowUpProposal(BaseModel):
    """What the reflector model proposed: what was kept, what was dropped, and the call itself."""

    accepted: list[SubQuestion] = Field(default_factory=list)
    rejected: list[Rejected] = Field(default_factory=list)
    call: ModelCall | None = None
    # What the model said about its proposal or its critique, in its own words: the card's
    # Reasoning (the reflector runs with thinking off, so the call itself carries none).
    reasoning: str = ""
    # The critique's verdict on whether the claims answer the question; None when the model
    # was asked for follow-ups rather than for a critique.
    sufficient: bool | None = None


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
        """The archetype's stop rule (Strategy), applied to this batch's verdicts."""
        observed = StopInput(verdicts=verdicts, entities_before=before, entities_after=after)
        return self._rule_for(archetype).decide(observed)

    def rule_name(self, archetype: Archetype) -> str:
        """The archetype's stop rule, by name, as a card says what decided."""
        return self._rule_for(archetype).name

    async def follow_ups(
        self, ctx: CheckContext, verdicts: list[CheckVerdict], tried: list[str], round_number: int
    ) -> list[SubQuestion]:
        """Model-proposed follow-ups, filtered to unmet items and untried queries."""
        return (await self.propose(ctx, verdicts, tried, round_number)).accepted

    async def propose(
        self, ctx: CheckContext, verdicts: list[CheckVerdict], tried: list[str], round_number: int
    ) -> FollowUpProposal:
        """The model's follow-ups, what of them was kept and dropped, and the call that made them."""
        gaps = [v for v in verdicts if not v.passed]
        if self._caller is None or not gaps or self._max == 0:
            return FollowUpProposal()
        answer, call = await self._caller.call_traced(
            self._prompt(ctx, gaps, tried), prompt.ReflectorAnswer
        )
        accepted, rejected = _sorted(
            answer, {v.item_id for v in gaps}, set(tried), round_number, self._max
        )
        return FollowUpProposal(
            accepted=accepted, rejected=rejected, call=call, reasoning=answer.rationale
        )

    async def critique(
        self, ctx: CheckContext, verdicts: list[CheckVerdict], tried: list[str], round_number: int
    ) -> FollowUpProposal:
        """Before the run stops: do the claims answer the question? If not, follow-ups for it."""
        items = set(ctx.contract.rubric.ids)
        if self._caller is None or not items:
            return FollowUpProposal()
        claims = [entry.claim for entry in ctx.ledger.entries]
        checks = [{"item_id": v.item_id, "passed": v.passed, "detail": v.detail} for v in verdicts]
        answer, call = await self._caller.call_traced(
            prompt.build_critique(ctx.contract, checks, tried, claims, self._max),
            prompt.CritiqueAnswer,
        )
        reasoning = answer.critique or answer.rationale
        if answer.sufficient:
            return FollowUpProposal(call=call, reasoning=reasoning, sufficient=True)
        accepted, rejected = _sorted(answer, items, set(tried), round_number, self._max)
        return FollowUpProposal(
            accepted=accepted,
            rejected=rejected,
            call=call,
            reasoning=reasoning,
            sufficient=False,
        )

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


def _sorted(
    answer: prompt.ReflectorAnswer,
    gap_ids: set[str],
    tried: set[str],
    round_number: int,
    limit: int,
) -> tuple[list[SubQuestion], list[Rejected]]:
    """The follow-ups that serve a gap with a fresh query, up to the limit; and the rest, with why."""
    accepted: list[SubQuestion] = []
    rejected: list[Rejected] = []
    seen = {t.casefold() for t in tried}
    for index, follow in enumerate(answer.follow_ups, start=1):
        serves = tuple(item for item in follow.rubric_items if item in gap_ids)
        fresh = tuple(query for query in follow.queries if query.casefold() not in seen)
        why = (
            NO_GAP
            if not serves
            else TRIED
            if not fresh
            else OVER_LIMIT
            if len(accepted) >= limit
            else ""
        )
        if why:
            rejected.append(Rejected(question=follow.question, why=why))
            continue
        accepted.append(
            SubQuestion(
                node_id=f"F{round_number}-{index}",
                question=follow.question,
                rubric_items=serves,
                queries=fresh,
                subject=follow.subject,
            )
        )
    return accepted, rejected
