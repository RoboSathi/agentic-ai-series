"""The gate's prompt: Q&A or Research, and for Research which kind."""

from typing import Literal

from pydantic import BaseModel

from research_agent.domain.scope import Archetype, Mode
from research_agent.prompts.base import RolePrompt

ROLE = "interviewer"

INSTRUCTIONS = """
Decide how to handle the user's first message.

- "scopable" with mode "qa" when it is a direct question a short, cited
  answer from a few web searches can settle: facts, definitions, recent
  news or developments, "what/who/when/how does X". Most questions are this.
- "scopable" with mode "research" when it asks for a decision between
  options, a comparison, a map of a market or field, verification of
  specific claims, a structured explanation of a subject, or explicitly
  asks for research or a report. Give the archetype: decision_support,
  landscape_brief, fact_verification, explainer, or general when none fits.
- "needs_reframe" only when the message contains no question and no
  purpose at all ("tell me about X").
- "refuse" when the request is harmful or targets a private person.

Name the topic in a few words (for qa: the question itself). If the
message states a time window ("last month", "this week", "past 3 days"),
give it in days as recent_days. Never start research here.
"""


class GateAnswer(BaseModel):
    decision: Literal["scopable", "needs_reframe", "refuse"]
    mode: Mode = "qa"
    archetype: Archetype | None = None
    topic: str = ""
    reason: str = ""
    recent_days: int | None = None


def build(message: str, cues: dict[str, list[str]], research_cues: list[str]) -> RolePrompt:
    """`cues` are phrasings typical of each Research kind, as classification hints."""
    return RolePrompt(
        role=ROLE,
        instructions=INSTRUCTIONS,
        payload={
            "task": "gate",
            "message": message,
            "archetype_cues": cues,
            "research_cues": research_cues,
        },
    )
