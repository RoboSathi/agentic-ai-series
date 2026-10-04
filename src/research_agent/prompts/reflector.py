"""The reflector's prompts: follow-ups for unmet items, and a critique once every check passes.

The reflector sees gaps (unmet rubric items with the check's findings) or, for the critique, the
whole rubric; what was already tried; and the ledger's claims — never a page.
"""

from typing import Any

from pydantic import BaseModel, Field

from research_agent.domain.rubric import Contract
from research_agent.prompts.base import RolePrompt
from research_agent.prompts.planner import brief_view

ROLE = "reflector"

INSTRUCTIONS = """
Some acceptance criteria are not yet met. Propose follow-up sub-questions
that could close them, each serving at least one of the listed gap ids.
Do not repeat queries that were already tried. Propose nothing for items
that are not listed as gaps. If nothing new could help, return an empty
list — an honest partial result is better than repeated searching.

Today's date and the approved recency window are in the payload (`today`,
`recency_days`). The search is already limited to that window: never put a
year, month or date in a query, and never rely on dates from memory.
"""


# Asked before every run stops on its own terms: the checks count citations, domains and terms;
# whether the claims actually answer the question is the model's judgement.
CRITIQUE_INSTRUCTIONS = """
The research is about to stop: either every acceptance criterion's automatic check passed, or
the rounds or budget ran out. `checks` lists each criterion and whether its check passed. Those
checks count citations, domains and matching terms; they cannot tell whether the evidence
actually answers the question. Judge that, and say why.

- The brief's topic is the question. Read it, the rubric and `claims_so_far`.
- `sufficient` is true when the claims answer the question as the brief asks it: specific, current
  as of `today`, and not contradicting each other on the main point. It is false when the central
  question is unanswered or answered only vaguely, when the claims are out of date for a question
  that names no date, or when the sources conflict on the main point.
- `critique`: two or three sentences on what is answered and what is missing or weak.
- When not sufficient, propose at most `max_follow_ups` follow-up sub-questions that would close
  what is missing, each naming the rubric item ids it strengthens and fresh queries — never one in
  `queries_tried`. When sufficient, propose none.

Never put a year, month or date in a query: the search already applies the brief's window.
"""


class FollowUp(BaseModel):
    question: str
    rubric_items: list[str] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)
    subject: str = ""


class ReflectorAnswer(BaseModel):
    follow_ups: list[FollowUp] = Field(default_factory=list)
    rationale: str = ""


class CritiqueAnswer(ReflectorAnswer):
    sufficient: bool = True
    critique: str = ""


def build(
    contract: Contract,
    gaps: list[dict[str, Any]],
    tried: list[str],
    claims: list[str],
    max_follow_ups: int,
) -> RolePrompt:
    return RolePrompt(
        role=ROLE,
        instructions=INSTRUCTIONS,
        payload={
            "task": "reflect",
            "today": contract.frozen_at.date().isoformat(),
            "recency_days": contract.brief.recency_days,
            "brief": brief_view(contract),
            "gaps": gaps,
            "queries_tried": tried,
            "claims_so_far": claims,
            "max_follow_ups": max_follow_ups,
        },
    )


def build_critique(
    contract: Contract,
    checks: list[dict[str, Any]],
    tried: list[str],
    claims: list[str],
    max_follow_ups: int,
) -> RolePrompt:
    """The critique: the whole rubric with each check's verdict, the claims, what was tried."""
    rubric = [{"item_id": item.item_id, "text": item.text} for item in contract.rubric.items]
    return RolePrompt(
        role=ROLE,
        instructions=CRITIQUE_INSTRUCTIONS,
        payload={
            "task": "critique",
            "today": contract.frozen_at.date().isoformat(),
            "recency_days": contract.brief.recency_days,
            "brief": brief_view(contract),
            "rubric": rubric,
            "checks": checks,
            "queries_tried": tried,
            "claims_so_far": claims,
            "max_follow_ups": max_follow_ups,
        },
    )
