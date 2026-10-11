"""The report says what was found, never how: the run's working notes do not reach the reader.

Regression for run-23252f6025 (2026-10-11), whose report told the reader "Sources E3–E7 are
from Facebook and Instagram reels." and "E1 and E2 are search snippets only".
"""

import re

import pytest

from research_agent.config import get_agent_settings
from research_agent.domain.report import Report, ReportSection
from research_agent.synthesis.citation_check import CitationCheck, strip_working_notes

pytestmark = pytest.mark.usefixtures("ra_config")

FINDINGS = (
    "Nightlife: The package mentions exciting nightlife as part of the experience [E3]. "
    "Quiet places in day: The evidence does not cover quiet places in day during daylight. "
    "Sources E3–E7 are from Facebook and Instagram reels. "
    "E1 and E2 are search snippets only [E1][E2]."
)
DETAILS = (
    "Package price is ₹8,499/- per person, according to a search result [E2]. "
    "Sources E1 and E2 are search snippets only and carry a caveat [E1][E2]."
)


def _strip(*bodies: str) -> CitationCheck:
    settings = get_agent_settings().synthesis
    patterns = [re.compile(p, re.IGNORECASE) for p in settings.working_note_patterns]
    report = Report(
        title="t",
        status="PARTIAL",
        sections=[ReportSection(heading=f"H{i}", body=b) for i, b in enumerate(bodies)],
    )
    check = CitationCheck(report=report)
    return strip_working_notes(check, patterns, re.compile(settings.citation_pattern))


def test_sentences_about_the_research_are_removed_and_counted() -> None:
    check = _strip(FINDINGS, DETAILS)
    findings, details = (section.body for section in check.report.sections)
    assert findings == (
        "Nightlife: The package mentions exciting nightlife as part of the experience [E3]."
    )
    # A hedge in the reader's words stays; a note about the evidence does not.
    assert details == "Package price is ₹8,499/- per person, according to a search result [E2]."
    assert len(check.removed) == 4


def test_a_citation_marker_alone_is_never_a_working_note() -> None:
    check = _strip("Hiring rose 4% in 2026 [E3][E12].")
    assert (
        check.report.sections[0].body == "Hiring rose 4% in 2026 [E3][E12]." and not check.removed
    )
