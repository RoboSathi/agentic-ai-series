"""The deterministic post-check: every citation in the report exists in the ledger.

A sentence citing an id the ledger never issued is a fabricated citation.
It is removed from the draft (and counted), so the delivered report can
only point at evidence that exists; the Goldfish Audit re-checks the same
rule independently on what is delivered.
"""

import re

from pydantic import BaseModel, Field

from agentic_commons.evidence import LedgerSnapshot
from research_agent.domain.report import Report, ReportSection

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class CitationCheck(BaseModel):
    report: Report
    removed: list[str] = Field(default_factory=list)
    unknown_ids: list[str] = Field(default_factory=list)


def cited_ids(report: Report, pattern: re.Pattern[str]) -> list[str]:
    text = "\n".join(section.body for section in report.sections)
    return list(dict.fromkeys(pattern.findall(text)))


def enforce_citations(
    report: Report, ledger: LedgerSnapshot, pattern: re.Pattern[str]
) -> CitationCheck:
    """The report with every sentence citing an unknown id removed."""
    known = {entry.entry_id for entry in ledger.entries}
    removed: list[str] = []
    sections = [_clean(section, known, pattern, removed) for section in report.sections]
    unknown = sorted({i for s in removed for i in pattern.findall(s) if i not in known})
    return CitationCheck(
        report=report.model_copy(update={"sections": sections}),
        removed=removed,
        unknown_ids=unknown,
    )


def _clean(
    section: ReportSection, known: set[str], pattern: re.Pattern[str], removed: list[str]
) -> ReportSection:
    kept_lines = []
    for line in section.body.splitlines():
        sentences = [s for s in _SENTENCE_END.split(line) if s]
        good = [s for s in sentences if set(pattern.findall(s)) <= known]
        removed.extend(s for s in sentences if s not in good)
        if good or not sentences:
            kept_lines.append(" ".join(good))
    return section.model_copy(update={"body": "\n".join(kept_lines).strip()})
