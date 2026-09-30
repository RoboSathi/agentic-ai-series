"""The synthesizer's prompt: write the report from the ledger, and only the ledger.

The payload carries ledger entries as `{entry_id, claim, source, date}`.
It has no field for a page and no field for a quoted span, so the model
writing the report never reads text a web page chose.
"""

from typing import Any

from pydantic import BaseModel, Field

from agentic_commons.evidence import LedgerSnapshot
from research_agent.domain.rubric import Contract
from research_agent.prompts.base import RolePrompt
from research_agent.prompts.planner import brief_view, rubric_view

ROLE = "synthesizer"

INSTRUCTIONS = """
Write the research report from the evidence ledger only.

- Use exactly the section headings given, in order.
- Every factual sentence cites at least one ledger entry as [E<n>]; cite
  only ids that appear in the ledger. Write no fact the ledger lacks.
- Where entries disagree, say so and cite both sides.
- For a decision, recommend an option and state the trade-offs.
- Do not write a Coverage & Limits, Known unknowns or Sources section;
  those are added from the run's records.
"""


class DraftSection(BaseModel):
    heading: str
    body: str


class SynthesisAnswer(BaseModel):
    title: str
    sections: list[DraftSection] = Field(default_factory=list)


def ledger_view(ledger: LedgerSnapshot) -> list[dict[str, Any]]:
    """Entries as the synthesizer sees them: claims and provenance, never spans."""
    return [
        {"entry_id": e.entry_id, "claim": e.claim, "source": e.source_url, "date": e.source_date}
        for e in ledger.entries
    ]


def build(
    contract: Contract, ledger: LedgerSnapshot, headings: list[str], gaps: list[str]
) -> RolePrompt:
    return RolePrompt(
        role=ROLE,
        instructions=INSTRUCTIONS,
        payload={
            "task": "synthesize",
            "brief": brief_view(contract),
            "rubric": rubric_view(contract),
            "ledger": ledger_view(ledger),
            "sections": headings,
            "unmet_items": gaps,
        },
    )
