"""The scopability gate's prompt: classify a first message into an archetype."""

from typing import Literal

from pydantic import BaseModel

from research_agent.domain.scope import Archetype
from research_agent.prompts.base import RolePrompt

ROLE = "interviewer"

INSTRUCTIONS = """
You decide whether a research request can be scoped. Classify the user's
first message:

- "scopable" with an archetype when it states or implies a purpose:
  decision_support (choose between options), landscape_brief (map players,
  approaches, trends), fact_verification (check specific claims), or
  explainer (explain how something works).
- "needs_reframe" when it is open-ended ("tell me about X") and states no
  decision, purpose or deliverable — the interview will reframe it.
- "refuse" when the request is harmful or targets a private person.

Name the topic in a few words. Never start research here.
"""


class GateAnswer(BaseModel):
    decision: Literal["scopable", "needs_reframe", "refuse"]
    archetype: Archetype | None = None
    topic: str = ""
    reason: str = ""


def build(message: str, cues: dict[str, list[str]]) -> RolePrompt:
    """`cues` are phrasings typical of each archetype, as classification hints."""
    return RolePrompt(
        role=ROLE,
        instructions=INSTRUCTIONS,
        payload={"task": "gate", "message": message, "archetype_cues": cues},
    )
