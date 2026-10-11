"""The synthesizer's prompt: write the report from the ledger, and only the ledger.

The payload carries ledger entries as `{entry_id, claim, source, date}`, plus a `caveat` on
an entry that rests on weaker evidence (a search snippet, a source outside the time window).
It has no field for a page and no field for a quoted span, so the model
writing the report never reads text a web page chose.
"""

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field

from agent_platform.evidence import LedgerSnapshot
from research_agent.domain.rubric import Contract
from research_agent.prompts.base import RolePrompt, dated
from research_agent.prompts.planner import brief_view, rubric_view

ROLE = "synthesizer"

INSTRUCTIONS = """
Write the research report from the evidence ledger only.

- Use exactly the section headings given, in order.
- Every factual sentence cites at least one ledger entry as [E<n>]; cite
  only ids that appear in the ledger. Write no fact the ledger lacks.
- Put the citation inside the sentence, before its final full stop:
  "Fact stated here [E1][E2]." — never after the stop ("Fact. [E1]").
- Be concise: a few short sentences per section, the most important findings first.
- Say no more than the claims you cite: keep their figures, units, names and
  qualifiers exactly as written ("Upto 1Cr" stays "up to 1 crore", never "₹1 crore
  for student startups" when the claim does not say who it is for).
- Where entries disagree, say so and cite both sides.
- An entry with a `caveat` is weaker evidence (a search snippet, or a source outside the
  time window): hedge the sentence that uses it in the reader's words ("according to a
  search result", "an older source reports"), and never state it as settled fact.
- Write about the subject, never about the research. Entry ids appear only inside the
  [E<n>] markers; never write about the ledger, entries, snippets, caveats, what kind of
  site a source is, or what the evidence does or does not cover. A part of the question
  the ledger does not cover is left out: the run adds a Coverage & Limits section for it.
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
        {
            "entry_id": e.entry_id,
            "claim": e.claim,
            "source": e.source_url,
            "date": e.source_date,
            **({"caveat": e.caveat} if e.caveat else {}),
        }
        for e in ledger.entries
    ]


def build(
    contract: Contract,
    ledger: LedgerSnapshot,
    headings: list[str],
    gaps: list[str],
    previous: list[dict[str, str]] | None = None,
    findings: list[dict[str, Any]] | None = None,
) -> RolePrompt:
    """The draft; after more research, also the previous report and what the audit found."""
    payload: dict[str, Any] = {
        "task": "synthesize",
        "brief": brief_view(contract),
        "rubric": rubric_view(contract),
        "ledger": ledger_view(ledger),
        "sections": headings,
        "unmet_items": gaps,
    }
    if previous:
        payload |= {"previous_report": previous, "findings": findings or []}
    request = RolePrompt(
        role=ROLE, instructions=INSTRUCTIONS, payload=dated(payload, contract.frozen_at.date())
    )
    return request.with_instruction(REDRAFT_INSTRUCTION) if previous else request


REVISE_INSTRUCTIONS = """
Revise the report in `previous_report` so it meets the audit's `findings`, from the evidence
ledger only.

- Each finding names a criterion the report failed, the reviewer's reason, and the sentences
  that drove the score. Change only what a finding asks for; keep every other sentence, with
  its citations, exactly as it is.
- Every factual sentence cites at least one ledger entry as [E<n>], inside the sentence before
  its full stop. Write no fact the ledger lacks: where a finding asks for something the ledger
  does not cover, leave it out (Coverage & Limits, added by the run, says what was not found).
- The findings are notes for you, not for the reader: never repeat their wording, the ids they
  name, or anything about entries, snippets, caveats or what the evidence covers.
- Use exactly the section headings given, in order.
- Do not write a Coverage & Limits, Known unknowns or Sources section.
- `skills_learned_from_past_runs` (when present) are skills a curator approved from what
  earlier runs' audits kept finding; follow them to avoid repeating a fault, never as evidence.
"""

# Appended to the drafting instructions when a report is drafted again after more research.
REDRAFT_INSTRUCTION = (
    "A previous report and the audit's findings on it are in `previous_report` and "
    "`findings`. Keep what the audit accepted, fix what it flagged, and use the new evidence."
)


def build_revision(
    contract: Contract,
    ledger: LedgerSnapshot,
    headings: list[str],
    previous: list[dict[str, str]],
    findings: list[dict[str, Any]],
    learned_skills: Sequence[str] = (),
) -> RolePrompt:
    """The revision: the report as it stands, what the audit found, and the ledger."""
    return RolePrompt(
        role=ROLE,
        instructions=REVISE_INSTRUCTIONS,
        payload=dated(
            {
                "task": "revise",
                "brief": brief_view(contract),
                "ledger": ledger_view(ledger),
                "sections": headings,
                "previous_report": previous,
                "findings": findings,
                **(
                    {"skills_learned_from_past_runs": list(learned_skills)}
                    if learned_skills
                    else {}
                ),
            },
            contract.frozen_at.date(),
        ),
    )
