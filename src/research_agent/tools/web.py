"""Web search and page fetch for the model, through the platform's `WebGateway`.

The gateway already routes DuckDuckGo then Tavily, caches, and enforces the free-tier quotas.
What the model reads says *when* it was retrieved and what date each source gives itself, so it
can tell a live snapshot from a final result. A fetched page is untrusted text: it is
phrase-screened (and model-screened when configured) and size-capped before the model sees it,
and the prompt tells the model it is data.
"""

import re
from datetime import datetime

from pydantic import BaseModel, Field

from agent_platform.logging import logged
from agent_platform.tools.types import ToolResult
from agent_platform.web import SearchRequest, WebGateway
from research_agent.config.settings import AnswerSettings
from research_agent.guardrails.injection import InjectionScreen
from research_agent.tools import PUBLISHED_KEY, QUERY_KEY, URLS_KEY

_BLANK_RUNS = re.compile(r"\n\s*\n+")


class SearchArgs(BaseModel):
    query: str = Field(description="What to search for.")
    recent_days: int | None = Field(
        default=None, ge=1, description="Only results from the last N days, when recency matters."
    )


class FetchArgs(BaseModel):
    url: str = Field(description="The page to read, taken from a search result.")


def _compact(text: str, limit: int) -> str:
    """Page text with its blank-line runs collapsed, cut to `limit` characters."""
    return _BLANK_RUNS.sub("\n", text.strip())[:limit]


def _retrieved(settings: AnswerSettings, when: datetime | None = None) -> str:
    stamp = (when or datetime.now()).astimezone().strftime(settings.date_format)
    return settings.messages.retrieved.format(when=stamp)


class WebSearch:
    def __init__(self, gateway: WebGateway, settings: AnswerSettings) -> None:
        self._gateway = gateway
        self._settings = settings

    @logged("research_agent")
    async def __call__(self, query: str, recent_days: int | None = None) -> ToolResult:
        notes = self._settings.messages
        response = await self._gateway.search(
            SearchRequest(
                query=query, max_results=self._settings.search_max_results, recent_days=recent_days
            )
        )
        if not response.hits:
            return ToolResult(success=True, output=notes.no_results)
        blocks = [_retrieved(self._settings)]
        for hit in response.hits:
            dated = notes.published.format(date=hit.published_date) if hit.published_date else ""
            snippet = _compact(hit.snippet, self._settings.snippet_max_chars)
            blocks.append("\n".join(part for part in (hit.title, hit.url, dated, snippet) if part))
        return ToolResult(
            success=True,
            output="\n\n".join(blocks),
            data={
                URLS_KEY: [hit.url for hit in response.hits],
                QUERY_KEY: query,
                PUBLISHED_KEY: [hit.published_date for hit in response.hits],
                "provider": response.provider,
            },
        )


class FetchUrl:
    def __init__(
        self, gateway: WebGateway, settings: AnswerSettings, screen: InjectionScreen | None = None
    ) -> None:
        self._gateway = gateway
        self._settings = settings
        self._screen = screen

    @logged("research_agent")
    async def __call__(self, url: str) -> ToolResult:
        page = await self._gateway.fetch(url)
        if not page.ok:
            return ToolResult(
                error=self._settings.messages.page_unreadable.format(status=page.status)
            )
        if self._screen is not None:
            verdict = await self._screen.assess(page.text)
            if verdict.drop:
                return ToolResult(
                    error=self._settings.messages.page_withheld.format(why=verdict.why)
                )
        notes = self._settings.messages
        dated = notes.published.format(date=page.published) if page.published else ""
        header = "\n".join(
            part
            for part in (_retrieved(self._settings, page.retrieved_at), page.url, dated)
            if part
        )
        body = _compact(page.text, self._settings.fetch_max_chars)
        return ToolResult(success=True, output=f"{header}\n\n{body}", data={URLS_KEY: [page.url]})
