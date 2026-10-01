"""The planner's prompt: turn the contract into a DAG of sub-questions.

The planner is shown the contract — never a page. Its payload is built
from `brief_view` (which leaves out the user's private context) and the
rubric; there is no field through which raw page text could arrive.
"""

from typing import Any

from pydantic import BaseModel, Field

from research_agent.domain.rubric import Contract
from research_agent.prompts.base import RolePrompt

ROLE = "planner"

INSTRUCTIONS = """
Plan the research as a small DAG of sub-questions.

- Every node must serve at least one rubric item (list its ids in
  `rubric_items`); a node that serves none will be rejected.
- Use `depends_on` only when a node genuinely needs another node's
  evidence first; independent nodes run in parallel.
- Give each node up to the allowed number of concrete web `queries`.
- Name the brief subject (an option, claim or concept) a node is about in
  `subject` when it is about one.
- Stay within the node limit. Do not plan work the rubric does not need.
- For a `qa` or `general` brief the topic is the question itself: plan the
  fewest nodes (often one) whose queries would find pages that answer it.
- When the brief has `recency_days`, write queries for recent coverage
  (the search is already limited to that window; do not add dates or
  words like "last month").
- Each node must cover a distinct angle. Never plan two nodes whose queries
  would return the same pages; for a broad question, one node with a specific
  query beats two near-duplicates. Name concrete subjects, not "AI news".
- Every query costs one search from `max_searches`. Plan at most
  `max_searches` minus one queries in total, so one search stays in reserve
  for a follow-up. Do not name publishers or sites unless the brief does.
"""

REPAIR_INSTRUCTION = "Your previous plan was rejected for these reasons; fix every one:\n{problems}"


class PlannedNode(BaseModel):
    node_id: str
    question: str
    depends_on: list[str] = Field(default_factory=list)
    rubric_items: list[str] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)
    subject: str = ""


class PlanAnswer(BaseModel):
    nodes: list[PlannedNode] = Field(default_factory=list)
    rationale: str = ""


def brief_view(contract: Contract) -> dict[str, Any]:
    """What a planning-side role may know of the Brief: everything but private context."""
    return contract.brief.model_dump(mode="json", exclude={"private_terms"})


def rubric_view(contract: Contract) -> list[dict[str, Any]]:
    return [
        {
            "item_id": item.item_id,
            "text": item.text,
            "must_pass": item.must_pass,
            "maps_to": list(item.maps_to),
        }
        for item in contract.rubric.items
    ]


def build(contract: Contract, max_nodes: int, max_queries: int) -> RolePrompt:
    return RolePrompt(
        role=ROLE,
        instructions=INSTRUCTIONS,
        payload={
            "task": "plan",
            "brief": brief_view(contract),
            "rubric": rubric_view(contract),
            "max_nodes": max_nodes,
            "max_queries_per_node": max_queries,
            "max_searches": contract.budget.max_searches,
        },
    )
