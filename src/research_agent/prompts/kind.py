"""The research-kind prompt: which kind of research a `/research` question asks for."""

from typing import Literal

from pydantic import BaseModel

from research_agent.domain.scope import Archetype
from research_agent.prompts.base import RolePrompt

ROLE = "interviewer"

INSTRUCTIONS = """
The user asked for a research run. Decide which kind of research it is.

- "scopable" with the archetype that fits:
  - decision_support: choosing between options, or a recommendation
  - landscape_brief: a map of a market, field or set of players
  - fact_verification: checking whether specific claims are true
  - explainer: a structured explanation of a subject or how something works
  - general: when none of these fits
- "refuse" when the request is harmful or targets a private person.

Name the topic in a few words. If the message states a time window ("last month", "this week",
"past 3 days"), give it in days as recent_days. `archetype_cues` are phrasings typical of each
kind, as hints. Do not answer the question.
"""


class KindAnswer(BaseModel):
    decision: Literal["scopable", "refuse"]
    archetype: Archetype | None = None
    topic: str = ""
    reason: str = ""
    recent_days: int | None = None


def build(message: str, cues: dict[str, list[str]]) -> RolePrompt:
    """`cues` are phrasings typical of each research kind, as classification hints."""
    return RolePrompt(
        role=ROLE,
        instructions=INSTRUCTIONS,
        payload={"task": "research_kind", "message": message, "archetype_cues": cues},
    )
