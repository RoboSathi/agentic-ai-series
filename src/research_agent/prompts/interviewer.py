"""The interviewer's prompt: one turn of the scoping interview."""

from datetime import date

from pydantic import BaseModel, Field

from research_agent.domain.scope import CoverageItem
from research_agent.prompts.base import RolePrompt, dated

ROLE = "interviewer"

INSTRUCTIONS = """
You are scoping a research request before any research starts. You keep a
coverage checklist; each item has a `goal` (what you need to learn) and is
open, covered or waived. The goal is for you, not for the user: never read
it out or quote it.

Each turn:
- Read the user's latest answer, and on the first turn the request itself.
  Record what it establishes in `updated_items` (item_id + value, in the
  user's own words). Never mark an item covered that nothing said addresses,
  and never ask what the request or an earlier answer already told you.
- Keep the interview as short as this request allows. An optional item
  (`required: false`) is worth a question only if the answer would change
  what is researched or how it is reported for THIS request; otherwise list
  it in `skip_items` with a short reason, and do not ask. An item marked
  `skippable: false` is never skipped: ask it, even though it is optional. A simple, narrow
  question needs few turns; a choice between options needs more.
- Challenge vague or absolute answers ("the best", "everything") to open
  questions (purpose, options, criteria, decisions) in `challenges`: ask
  what would make one option better than another, or what would change the
  decision. Never challenge an answer to an item that has `choices`: a
  picked choice is a complete answer.
- Ask about the subject of THIS request (`request`) before anything about its
  use: which parts of the topic matter — its sub-areas, the named options,
  players, products, regions or time frame — with examples taken from the
  topic itself ("models, policy, funding or products?"). Ask about purpose,
  format or depth only when the request does not already imply it; when it
  does, record the implied value in `updated_items` instead of asking.
- Ask exactly one `next_question`, about the most important open item
  (`next_item_id`). Write it fresh, for this request: one short, plain
  sentence (about twenty words or fewer), no jargon, no checklist words,
  and where it helps, a concrete example drawn from the user's own topic.
  When the item has `choices`, a menu of them is shown beside your question,
  so ask a question those choices answer, and do not list them yourself.
  Never tell the user what to type.
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


class ItemSkip(BaseModel):
    """An optional item the interviewer judged irrelevant to this request."""

    item_id: str
    reason: str = ""


class InterviewTurn(BaseModel):
    """One structured interviewer turn."""

    next_question: str = ""
    next_item_id: str = ""
    challenges: list[str] = Field(default_factory=list)
    updated_items: list[ItemUpdate] = Field(default_factory=list)
    skip_items: list[ItemSkip] = Field(default_factory=list)
    proposes_done: bool = False
    draft_plan: list[str] = Field(default_factory=list)


def build(
    *,
    archetype: str,
    request: str,
    topic: str,
    items: list[CoverageItem],
    asked_item_id: str,
    answer: str,
    turn: int,
    today: date,
) -> RolePrompt:
    return RolePrompt(
        role=ROLE,
        instructions=INSTRUCTIONS,
        payload=dated(
            {
                "task": "interview",
                "archetype": archetype,
                "request": request,
                "topic": topic,
                "turn": turn,
                "asked_item_id": asked_item_id,
                "answer": answer,
                "checklist": [_visible(item) for item in items],
            },
            today,
        ),
    )


def _visible(item: CoverageItem) -> dict[str, object]:
    """An item as the model sees it; a private value is replaced by a marker."""
    value = "(private, recorded)" if item.private and item.value else item.value
    if item.status != "open":
        # Settled: what it holds is all the model needs, not its goal and options again.
        return {"item_id": item.item_id, "status": item.status, "value": value}
    return {
        "item_id": item.item_id,
        "goal": item.goal,
        "choices": [choice.label for choice in item.choices],
        "required": item.required,
        "skippable": item.skippable,
        "status": item.status,
        "value": value,
        "private": item.private,
    }
