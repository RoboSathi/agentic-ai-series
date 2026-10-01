"""The router's fallback prompt: answer directly, or start a research run."""

from typing import Literal

from pydantic import BaseModel

from research_agent.prompts.base import RolePrompt

ROLE = "interviewer"

INSTRUCTIONS = """
Decide how to handle the user's message.

- "answer" when it is a question a short answer settles, with or without a
  quick web lookup: facts, definitions, current conditions, recent news,
  arithmetic, "what/who/when/how does X". Most messages are this.
- "research" only when it asks for a decision between options, a comparison, a
  map of a market or field, verification of several claims, a structured
  explanation of a whole subject, or explicitly asks for research or a report.

Give a one-line reason.
""".strip()


class IntentAnswer(BaseModel):
    label: Literal["answer", "research"]
    reason: str = ""


def build(message: str, candidates: list[dict[str, object]]) -> RolePrompt:
    """`candidates` are the encoder's labels and scores, as a hint it was unsure about."""
    return RolePrompt(
        role=ROLE,
        instructions=INSTRUCTIONS,
        payload={"task": "route", "message": message, "encoder_candidates": candidates},
    )
