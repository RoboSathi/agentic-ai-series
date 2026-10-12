"""The judge's prompt: score one judgement-only criterion against the report.

The judge sees the criterion, the report and the ledger's claims — the
independent audit's inputs, nothing from the conversation that produced them.
"""

from datetime import date

from pydantic import BaseModel, Field

from agent_platform.hallucination_check.evidence import LedgerSnapshot
from research_agent.domain.report import Report
from research_agent.prompts.base import RolePrompt, dated

ROLE = "judge"

INSTRUCTIONS = """
You are an independent reviewer. You did not write this report and have
no stake in it. Score how well the report meets the criterion, from 0.0
(not at all) to 1.0 (fully), using only the report and the listed
evidence. Explain the score in one sentence; list the sentences that
drove it.
"""


class JudgeAnswer(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    rationale: str = ""
    evidence: list[str] = Field(default_factory=list)


def build(
    criterion: str, check: str, report: Report, ledger: LedgerSnapshot, today: date
) -> RolePrompt:
    return RolePrompt(
        role=ROLE,
        instructions=INSTRUCTIONS,
        payload=dated(
            {
                "task": "judge",
                "criterion": criterion,
                "check": check,
                "report": report.text(),
                "evidence": [{"entry_id": e.entry_id, "claim": e.claim} for e in ledger.entries],
            },
            today,
        ),
    )
