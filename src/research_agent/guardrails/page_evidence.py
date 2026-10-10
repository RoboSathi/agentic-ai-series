"""What a direct answer is grounded on: the content its sources actually hold.

The loop's observations are the first of it: every page the model fetched, and every search
hit as its own source. But an answer is often written from search snippets alone — a few
hundred characters cut from a page — and then cites the page. Its claims are checked against
the page: each cited URL the loop saw only as a snippet is fetched here, through the same
gateway and injection screen as the model's own fetches, and read in full.

Bounded by `cited_fetch_max` pages and `cited_fetch_seconds` (never more than the turn has
left). A page that cannot be read in time leaves its snippet as the evidence.
"""

import asyncio
import re
from collections.abc import Awaitable, Callable

from agent_platform.context.types import Chunk
from agent_platform.hallucination_check import Observation, tool_evidence
from agent_platform.logging import get_logger, instrument
from agent_platform.tools.types import ToolResult
from research_agent.config.settings import AnswerGroundingSettings

# Reads one page as the model's own fetch does (`tools/web.py::FetchUrl`): gateway, screen, cap.
PageReader = Callable[[str], Awaitable[ToolResult]]

# A URL as an answer writes it, without the punctuation that closes the sentence or the link.
_URL = re.compile(r"https?://[^\s)\]>\"'`]+")
_TRAILING = ".,;:!?"


@instrument("research_agent")
class PageEvidence:
    """The observations, plus the full page of every cited URL seen only as a snippet."""

    def __init__(self, fetch: PageReader, settings: AnswerGroundingSettings) -> None:
        self._fetch = fetch
        self._max = settings.cited_fetch_max
        self._seconds = settings.cited_fetch_seconds
        self._logger = get_logger("research_agent")

    async def gather(
        self, answer: str, observations: list[Observation], pages: list[str], seconds_left: float
    ) -> list[Chunk]:
        """Every premise the answer may be checked against, one tool result each."""
        wanted = self._snippet_only(answer, observations, pages)
        fetched = await self._read(wanted, min(self._seconds, seconds_left)) if wanted else []
        return tool_evidence([*observations, *fetched])

    def _snippet_only(
        self, answer: str, observations: list[Observation], pages: list[str]
    ) -> list[str]:
        """The answer's cited URLs that the loop never read in full, first cited first."""
        read = set(pages)
        cited = dict.fromkeys(url.rstrip(_TRAILING) for url in _URL.findall(answer))
        return [url for url in cited if url not in read][: self._max]

    async def _read(self, urls: list[str], seconds: float) -> list[Observation]:
        """The pages that could be read in time; the rest keep only their snippets."""
        if seconds <= 0:
            return []
        tasks = [asyncio.ensure_future(self._fetch(url)) for url in urls]
        done, pending = await asyncio.wait(tasks, timeout=seconds)
        for task in pending:
            task.cancel()
        if pending:
            late = [url for url, task in zip(urls, tasks, strict=True) if task in pending]
            self._logger.warning("Cited pages not read in time", urls=late, seconds=seconds)
        return [
            observation
            for url, task in zip(urls, tasks, strict=True)
            if task in done and (observation := self._observed(url, task)) is not None
        ]

    def _observed(self, url: str, task: "asyncio.Future[ToolResult]") -> Observation | None:
        """A page read in full, as an observation; a failed read is logged and grounds nothing."""
        error = task.exception()
        result = None if error is not None else task.result()
        if result is not None and result.success:
            self._logger.info(
                "Cited page read for grounding",
                url=url,
                chars=len(result.output),
                page=result.output,
            )
            return Observation(result.output, source=url)
        why = result.error if result is not None else repr(error)
        self._logger.warning("Cited page unreadable for grounding", url=url, error=why)
        return None


__all__ = ["PageEvidence", "PageReader"]
