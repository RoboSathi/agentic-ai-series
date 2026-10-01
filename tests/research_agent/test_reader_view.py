"""The reader view, on the report from a real Q&A run ("Important AI news in the last 1 month")."""

import re
from datetime import datetime, timezone

from research_agent.domain.report import Citation, Report, ReportSection
from research_agent.synthesis.reader_view import render_reader_view, split_findings

PBS = "https://www.pbs.org/newshour/world/contradicting-trump-pope-leo-says-ai-safety-concerns-not-fake-news"
PRN = "https://www.prnewswire.com/news-releases/omega-healthcare-receives-geekwire-and-uipath-ai-breakthrough-award-302893014.html"
BIG = "https://bigthink.com/starts-with-a-bang/ai-first-breakthrough-theoretical-physics"
WHEN = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)
PATTERN = r"\[(E\d+)\]"
HIDDEN = {"Known unknowns", "Rejected sources", "Sources", "Coverage & Limits"}
LABELS = {"press release": ["prnewswire.com", "businesswire.com"]}

ANSWER = (
    "Pope Leo XIV stated that concerns about AI going rogue are not 'fake news' [E1], while "
    "President Donald Trump characterized those concerns as a 'hoax' [E2]. Anthropic CEO Dario "
    "Amodei warned that AI could lead a swarm of agents [E3]. Omega Healthcare's agent reduced "
    "handling time by up to 75% [E4]. The AI Breakthrough Award was created to recognize "
    "customers [E5]. AI achieved its first meaningful breakthrough in theoretical physics, with "
    "an AI model surpassing the previous human limit [E6][E7]. Claude Fable, running version 5.1, "
    "achieved theoretical results [E8]."
)
URLS = {"E1": PBS, "E2": PBS, "E3": PBS, "E4": PRN, "E5": PRN, "E6": BIG, "E7": BIG, "E8": BIG}


def _report(**overrides: object) -> Report:
    base = {
        "title": "Important AI News in the Last Month",
        "status": "PARTIAL",
        "sections": [
            ReportSection(heading="Answer", body=ANSWER),
            ReportSection(heading="Known unknowns", body="- None declared during scoping."),
            ReportSection(heading="Coverage & Limits", body="- Every must-pass criterion was met."),
        ],
        "citations": [
            Citation(entry_id=e, url=u, span="s", page_hash="h", retrieved_at=WHEN)
            for e, u in URLS.items()
        ],
        "rejected_codes": {
            "https://apnews.com/article/x": "blocked",
            "https://www.bloomberg.com/news/y": "blocked",
            "https://www.wsj.com/z": "blocked",
            "https://www.wired.com/story/w": "error",
        },
    }
    return Report(**{**base, **overrides})


def _view(report: Report) -> str:
    return render_reader_view(
        report, citation_pattern=PATTERN, hidden_headings=HIDDEN, source_labels=LABELS
    )


def test_same_page_entries_collapse_to_one_source_number() -> None:
    text = _view(_report())
    assert "[E" not in text
    assert (
        "• Pope Leo XIV stated that concerns about AI going rogue are not 'fake news' [1]" in text
    )
    assert "swarm of agents [1]" in text
    assert "up to 75% [2]" in text
    assert "human limit [3]" in text  # [E6][E7] -> a single [3]
    assert "[3][3]" not in text


def test_every_finding_is_its_own_bullet() -> None:
    text = _view(_report())
    assert sum(line.startswith("• ") for line in text.splitlines()) == 6  # E1+E2 share a sentence


def test_sources_list_resolves_every_number() -> None:
    text = _view(_report())
    assert "[1] pbs.org — " + PBS + " — retrieved 2026-10-01" in text
    assert "[2] prnewswire.com (press release) — " in text
    assert "[3] bigthink.com — " in text


def test_empty_bookkeeping_is_hidden_and_unread_pages_are_grouped() -> None:
    text = _view(_report())
    assert "Known unknowns" not in text and "Coverage & Limits" not in text
    assert (
        "**Not read:** apnews.com (blocked or paywalled), bloomberg.com (blocked or paywalled)"
        in text
    )
    assert "wired.com (could not be fetched)" in text
    assert "https://apnews.com" not in text  # domains, not URLs


def test_a_run_with_no_findings_says_so_and_what_was_not_established() -> None:
    from research_agent.domain.report import LimitEntry

    report = _report(
        sections=[ReportSection(heading="Answer", body="")],
        citations=[],
        limits=[
            LimitEntry(item_id="R6", text="At least 1 cited finding.", tried_queries=["a", "b"])
        ],
        budget_notes=["searches exhausted"],
    )
    text = _view(report)
    assert "No usable evidence was found." in text
    assert "Not established" in text and "Tried: a, b." in text
    assert "Stopped early: searches exhausted." in text
    assert "Sources" not in text


def test_unknown_evidence_ids_are_dropped_not_shown() -> None:
    report = _report(sections=[ReportSection(heading="Answer", body="A claim [E1][E99].")])
    assert "[1]" in _view(report) and "E99" not in _view(report)


def test_a_citation_ends_a_finding_not_every_full_stop() -> None:
    pattern = __import__("re").compile(PATTERN)
    body = "U.S. officials said the rule starts in May [E1]. A second claim [E2]."
    assert split_findings(body, pattern) == [
        "U.S. officials said the rule starts in May [E1].",
        "A second claim [E2].",
    ]


def test_citations_after_the_full_stop_stay_with_their_sentence() -> None:
    """The model writes `Fact. [E1] Next fact. [E2]`; each marker belongs to the sentence before."""
    from research_agent.synthesis.citation_check import normalize_citations

    pattern = re.compile(r"\[(E\d+)\]")
    body = "A is core. [E1][E2] B too. [E5] C last. [E6]"
    assert normalize_citations(body, pattern) == "A is core [E1][E2]. B too [E5]. C last [E6]."
    # The reader view copes with the old shape too: no bullet is a bare marker.
    findings = split_findings(body, pattern)
    assert findings == ["A is core. [E1][E2]", "B too. [E5]", "C last. [E6]"]
