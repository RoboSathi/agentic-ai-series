"""The reflector's prompt: propose follow-up research for unmet items only.

The reflector sees gaps (unmet rubric items with the check's findings),
what was already tried, and the ledger's claims — never a page.
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


class FollowUp(BaseModel):
    question: str
    rubric_items: list[str] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)
    subject: str = ""


class ReflectorAnswer(BaseModel):
    follow_ups: list[FollowUp] = Field(default_factory=list)
    rationale: str = ""


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
