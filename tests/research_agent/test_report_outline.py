"""The report's own outline, held to its limits in code (`synthesis/outline.py`)."""

from research_agent.domain.report import COVERAGE_AND_LIMITS, ReportSection
from research_agent.synthesis.outline import fit_outline

RESERVED = {"Sources", COVERAGE_AND_LIMITS}
FOLDED = "**{heading}**"


def _s(heading: str, body: str = "A finding [E1].") -> ReportSection:
    return ReportSection(heading=heading, body=body)


def test_the_writers_headings_are_kept_in_its_order() -> None:
    sections = [_s("Cycle life under daily use"), _s("What wears the cells")]
    assert fit_outline(sections, RESERVED, 4, FOLDED) == sections


def test_reserved_and_empty_sections_are_dropped() -> None:
    sections = [_s("Answer"), _s("sources", "- a.example"), _s("Gaps", "  ")]
    assert [s.heading for s in fit_outline(sections, RESERVED, 4, FOLDED)] == ["Answer"]


def test_a_heading_used_twice_is_one_section() -> None:
    fitted = fit_outline([_s("Cost", "One [E1]."), _s("cost", "Two [E2].")], RESERVED, 4, FOLDED)
    assert len(fitted) == 1 and "One [E1]." in fitted[0].body and "Two [E2]." in fitted[0].body


def test_sections_past_the_limit_fold_into_the_last_kept() -> None:
    sections = [_s("A"), _s("B"), _s("C", "Third [E3]."), _s("D", "Fourth [E4].")]
    fitted = fit_outline(sections, RESERVED, 2, FOLDED)
    assert [s.heading for s in fitted] == ["A", "B"]
    assert "**C**\nThird [E3]." in fitted[1].body and "**D**\nFourth [E4]." in fitted[1].body
