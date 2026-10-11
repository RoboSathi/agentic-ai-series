"""The deterministic post-check: every citation in the report exists in the ledger.

A sentence citing an id the ledger never issued is a fabricated citation.
It is removed from the draft (and counted), so the delivered report can
only point at evidence that exists; the Independent Audit re-checks the same
rule independently on what is delivered.
"""

import re
from collections.abc import Callable, Sequence

from pydantic import BaseModel, Field

from agent_platform.evidence import LedgerSnapshot
from agent_platform.observability.step_card import ModelCall
from research_agent.domain.report import Report, ReportSection

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class CitationCheck(BaseModel):
    report: Report
    removed: list[str] = Field(default_factory=list)
    unknown_ids: list[str] = Field(default_factory=list)
    # The synthesizer's call that wrote the draft (None when there was nothing to write from).
    call: ModelCall | None = None


def normalize_citations(body: str, pattern: re.Pattern[str]) -> str:
    """Citations moved to before the sentence's full stop, wherever the model put them.

    A model often writes `A fact. [E1][E2] Next fact. [E3]`: the marker follows the stop, so
    every splitter that ends a sentence at its stop attaches it to the *next* sentence, and
    the audit then checks each sentence against the wrong evidence. One convention, here,
    before anything reads the body: `A fact [E1][E2]. Next fact [E3].`
    """
    run = rf"(?:{pattern.pattern})(?:\s*(?:{pattern.pattern}))*"
    misplaced = re.compile(rf"(?P<stop>[.!?])[ \t]*(?P<cites>{run})")
    return misplaced.sub(lambda m: f" {_compact(m.group('cites'))}{m.group('stop')}", body)


def _compact(citations: str) -> str:
    """A citation run without the whitespace between its markers."""
    return re.sub(r"\s+", "", citations)


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


def strip_working_notes(
    check: CitationCheck, patterns: Sequence[re.Pattern[str]], citation: re.Pattern[str]
) -> CitationCheck:
    """The report without sentences about the research itself, each one counted as removed.

    The writer reads ledger entries, their ids and their caveats; a sentence that talks about
    those ("Sources E3–E7 are from reels", "the evidence does not cover X") is the run's
    working notes, not a finding. Matched on the sentence with its citation markers removed,
    so `[E3]` itself never matches; Coverage & Limits, built in code, says what was not found.
    """
    if not patterns:
        return check
    removed: list[str] = []

    def note(sentence: str) -> bool:
        plain = citation.sub("", sentence)
        return any(pattern.search(plain) for pattern in patterns)

    sections = [_kept(section, note, removed) for section in check.report.sections]
    return check.model_copy(
        update={
            "report": check.report.model_copy(update={"sections": sections}),
            "removed": [*check.removed, *removed],
        }
    )


def _kept(section: ReportSection, drop: Callable[[str], bool], removed: list[str]) -> ReportSection:
    kept_lines = []
    for line in section.body.splitlines():
        sentences = [s for s in _SENTENCE_END.split(line) if s]
        good = [s for s in sentences if not drop(s)]
        removed.extend(s for s in sentences if s not in good)
        if good or not sentences:
            kept_lines.append(" ".join(good))
    return section.model_copy(update={"body": "\n".join(kept_lines).strip()})


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
