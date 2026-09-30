"""The interviewer's prompt: one turn of the scoping interview."""

from pydantic import BaseModel, Field

from research_agent.domain.scope import CoverageItem
from research_agent.prompts.base import RolePrompt

ROLE = "interviewer"

INSTRUCTIONS = """
You are scoping a research request before any research starts. You keep a
coverage checklist; each item is open, covered or waived.

Each turn:
- Read the user's latest answer. Record what it establishes in
  `updated_items` (item_id + value, in the user's own words). Never mark
  an item covered that the answer does not address.
- Challenge vague or absolute answers ("the best", "everything") in
  `challenges`: ask what would make one option better than another, or
  what would change the decision. Do not simply agree.
- Ask exactly one `next_question`, about the most important open item
  (`next_item_id`).
- Set `proposes_done` only when no required item is open, and then write
  a short `draft_plan`: the sub-questions you would research. You propose
  the plan; do not ask the user to write it.
- Private items (marked private) are recorded but never repeated back.
"""

CHALLENGE_INSTRUCTION = (
    "Your last turns only agreed with the user. This turn MUST include at least one "
    "challenge: question an assumption, a vague answer, or a missing criterion."
)
REFRAME_INSTRUCTION = (
    "The request is open-ended. Ask what the research is for: choosing between options, "
    "mapping a field, checking specific claims, or explaining how something works."
)


class ItemUpdate(BaseModel):
    item_id: str
    value: str


class InterviewTurn(BaseModel):
    """One structured interviewer turn."""

    next_question: str = ""
    next_item_id: str = ""
    challenges: list[str] = Field(default_factory=list)
    updated_items: list[ItemUpdate] = Field(default_factory=list)
    proposes_done: bool = False
    draft_plan: list[str] = Field(default_factory=list)


def build(
    *,
    archetype: str,
    topic: str,
    items: list[CoverageItem],
    asked_item_id: str,
    answer: str,
    turn: int,
) -> RolePrompt:
    return RolePrompt(
        role=ROLE,
        instructions=INSTRUCTIONS,
        payload={
            "task": "interview",
            "archetype": archetype,
            "topic": topic,
            "turn": turn,
            "asked_item_id": asked_item_id,
            "answer": answer,
            "checklist": [_visible(item) for item in items],
        },
    )


def _visible(item: CoverageItem) -> dict[str, object]:
    """An item as the model sees it; a private value is replaced by a marker."""
    value = "(private, recorded)" if item.private and item.value else item.value
    return {
        "item_id": item.item_id,
        "prompt": item.prompt,
        "required": item.required,
        "status": item.status,
        "value": value,
        "private": item.private,
    }
