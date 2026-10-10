"""The deliverable: a report written from the ledger, and how the run ended."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

RunStatus = Literal["COMPLETE", "PARTIAL", "CANCELLED", "FAILED"]
SessionPhase = Literal[
    "SCOPING", "RUBRIC_REVIEW", "RUNNING", "AUDITING", "STOPPED", "DELIVERED", "CANCELLED", "FAILED"
]
COVERAGE_AND_LIMITS = "Coverage & Limits"


class Citation(BaseModel):
    """Enough to re-check a claim without the (discarded) ledger."""

    model_config = ConfigDict(frozen=True)

    entry_id: str
    url: str
    span: str
    page_hash: str
    retrieved_at: datetime
    # Why the entry is weaker evidence ("search snippet only"), or "" for a page read in full.
    caveat: str = ""


class ReportSection(BaseModel):
    """One heading and its prose; citations appear inline as `[E3]`."""

    heading: str
    body: str


class LimitEntry(BaseModel):
    """One unmet must-pass item: what it was, what was tried, why the run stopped."""

    item_id: str
    text: str
    tried_queries: list[str] = Field(default_factory=list)
    sources_examined: list[str] = Field(default_factory=list)
    reason: str = ""


class Report(BaseModel):
    """The research report."""

    title: str
    status: RunStatus
    sections: list[ReportSection] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    limits: list[LimitEntry] = Field(default_factory=list)
    rejected_sources: list[str] = Field(default_factory=list)
    # Why each rejected URL was not used (a short code), and any cap that cut the run short.
    rejected_codes: dict[str, str] = Field(default_factory=dict)
    budget_notes: list[str] = Field(default_factory=list)
    known_unknowns: list[str] = Field(default_factory=list)
    # Said above the findings when they rest on weaker evidence than the run was meant to
    # gather (a best-effort report written from search results alone); "" otherwise.
    notice: str = ""

    def section(self, heading: str) -> ReportSection | None:
        return next((section for section in self.sections if section.heading == heading), None)

    def text(self) -> str:
        return "\n\n".join(
            f"## {section.heading}\n{section.body.strip()}".rstrip() for section in self.sections
        ).strip()
