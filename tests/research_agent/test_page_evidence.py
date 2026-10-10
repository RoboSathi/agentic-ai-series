"""A direct answer is grounded on the content behind it, not on snippets of it.

The RoboSathi answer came back "4 unverified of 4": its evidence was the fetched page, but cut
into single lines no claim could be entailed by. These pin the evidence the check now reads.
"""

import asyncio
from typing import Any

import pytest

from agent_platform.hallucination_check import PASSAGE, Observation
from agent_platform.tools.types import ToolResult
from research_agent.config import get_agent_settings
from research_agent.graph.answer_nodes import _observations
from research_agent.guardrails.page_evidence import PageEvidence
from research_agent.tools import HITS_KEY, URLS_KEY

pytestmark = pytest.mark.usefixtures("ra_config")

PAGE = "https://robosathi.com/about/"
OTHER = "https://robosathi.com/courses/"


class _Pages:
    """Reads a page by URL: text, an error, or a page that never answers."""

    def __init__(self, pages: dict[str, Any]) -> None:
        self._pages = pages
        self.read: list[str] = []

    async def __call__(self, url: str) -> ToolResult:
        self.read.append(url)
        what = self._pages[url]
        if what is None:
            await asyncio.sleep(60)
        if isinstance(what, str):
            return ToolResult(success=True, output=what)
        return ToolResult(error="HTTP 403")


def _evidence(pages: dict[str, Any], **overrides: Any) -> tuple[PageEvidence, _Pages]:
    settings = get_agent_settings().answer.grounding.model_copy(update=overrides)
    reader = _Pages(pages)
    return PageEvidence(reader, settings), reader


async def test_a_cited_page_seen_only_as_a_snippet_is_read_in_full() -> None:
    evidence, reader = _evidence({PAGE: "Kumar Karan\nLead Instructor @ Scaler Academy"})
    answer = f"It is taught by Kumar Karan.\n\nSources:\n- {PAGE}"
    chunks = await evidence.gather(answer, [Observation("a snippet", source=PAGE)], [], 30)
    assert reader.read == [PAGE]
    assert {c.content for c in chunks if c.source == PAGE} >= {
        "Kumar Karan",
        "Lead Instructor @ Scaler Academy",
    }


async def test_a_page_the_loop_already_fetched_is_not_fetched_again() -> None:
    evidence, reader = _evidence({PAGE: "unused"})
    await evidence.gather(f"See {PAGE}.", [Observation("the page", source=PAGE)], [PAGE], 30)
    assert reader.read == []


async def test_at_most_the_configured_number_of_pages_are_read() -> None:
    evidence, reader = _evidence({PAGE: "a", OTHER: "b"}, cited_fetch_max=1)
    await evidence.gather(f"{PAGE} and {OTHER}", [], [], 30)
    assert reader.read == [PAGE]


async def test_a_page_that_cannot_be_read_leaves_the_snippet_as_the_evidence() -> None:
    evidence, _ = _evidence({PAGE: 403})
    chunks = await evidence.gather(f"{PAGE}", [Observation("the snippet", source=PAGE)], [], 30)
    assert [c.content for c in chunks if not c.metadata.get(PASSAGE)] == ["the snippet"]


async def test_a_slow_page_is_abandoned_and_a_quick_one_kept() -> None:
    evidence, _ = _evidence({PAGE: "Quick page.", OTHER: None}, cited_fetch_seconds=0.2)
    chunks = await evidence.gather(f"{PAGE} {OTHER}", [], [], 30)
    assert [c.content for c in chunks] == ["Quick page."]


async def test_no_time_left_reads_nothing() -> None:
    evidence, reader = _evidence({PAGE: "a"})
    await evidence.gather(PAGE, [], [], 0)
    assert reader.read == []


def test_a_search_is_one_observation_per_hit_each_from_its_own_page() -> None:
    result = ToolResult(
        success=True,
        output="all of it",
        data={URLS_KEY: [PAGE, OTHER], HITS_KEY: ["about text", "courses text"]},
    )
    seen = _observations("answer__web_search", "all of it", result, [PAGE, OTHER])
    assert [(o.source, o.text) for o in seen] == [(PAGE, "about text"), (OTHER, "courses text")]


def test_a_failed_call_grounds_nothing() -> None:
    assert _observations("answer__fetch_url", "error", ToolResult(error="x"), []) == []
