"""Search providers (Strategy): who answers "which pages mention this?".

Two implementations: `DuckDuckGoProvider` (free, no key, no official API —
best effort) and `TavilyProvider` (keyed, metered, supports time ranges and
a news vertical). `SearchRouter` (router.py) decides which is asked.

`TavilyProvider` Its wire shape was taken from Tavily's
published API reference (`POST https://api.tavily.com/search`, bearer
auth; response `{query, answer, results[{title, url, content, score,
raw_content, published_date}], response_time, request_id}`); every field
but `results[].url` is treated as optional, because an adapter that
crashes on a missing teaser has made the teaser load-bearing.

`answer` is mapped to `provider_summary` and nothing else: it is text a
model wrote about pages nobody here has read, so it never becomes a page.
"""

import asyncio
from collections.abc import Callable, Mapping
from typing import Any, Protocol, runtime_checkable

import httpx

from agentic_commons.config.settings import DuckDuckGoSettings, TavilySettings
from agentic_commons.resilience.rate_limiter.limiter import RateLimiter
from agentic_commons.web.types import SearchHit, SearchRequest, SearchResponse

_PROVIDER = "tavily"
_DUCKDUCKGO = "duckduckgo"
_NEWS = "news"


def window_name(days: int, table: Mapping[int, str]) -> str:
    """The narrowest named window that still covers `days` (the widest if none does)."""
    covering = [limit for limit in table if limit >= days]
    return table[min(covering)] if covering else table[max(table)]


@runtime_checkable
class SearchProvider(Protocol):
    """Answers one search request."""

    @property
    def name(self) -> str: ...

    async def search(self, request: SearchRequest) -> SearchResponse: ...


class MissingCredentialError(RuntimeError):
    """A live provider was asked to search with no API key configured."""


class TavilyProvider:
    """Tavily's REST search endpoint, through an injected (rate-limited) client."""

    def __init__(
        self, settings: TavilySettings, client: httpx.AsyncClient, api_key: str | None = None
    ) -> None:
        """`api_key` overrides the environment — for a recorder talking to a stand-in API."""
        self._settings = settings
        self._client = client
        self._api_key = api_key

    @property
    def name(self) -> str:
        return _PROVIDER

    async def search(self, request: SearchRequest) -> SearchResponse:
        key = self._api_key if self._api_key is not None else self._settings.api_key
        if not key:
            raise MissingCredentialError("tavily_api_key is not set")
        response = await self._client.post(
            f"{self._settings.base_url.rstrip('/')}{self._settings.endpoint}",
            json=self._body(request),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            timeout=self._settings.timeout_seconds,
        )
        response.raise_for_status()
        return parse_tavily_response(request, response.json())

    def _body(self, request: SearchRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "query": request.query,
            "max_results": request.max_results,
            "search_depth": self._settings.search_depth,
            "include_answer": self._settings.include_answer,
            "include_raw_content": False,
        }
        if request.include_domains:
            body["include_domains"] = list(request.include_domains)
        if request.exclude_domains:
            body["exclude_domains"] = list(request.exclude_domains)
        if request.recent_days is not None:
            body["time_range"] = window_name(request.recent_days, self._settings.time_ranges)
        if request.topic:
            body["topic"] = request.topic
        return body


SearchBackend = Callable[[str, SearchRequest], list[dict[str, Any]]]


class DuckDuckGoProvider:
    """DuckDuckGo via the `ddgs` package, run off the event loop.

    There is no official API and no key, so this is the cheap first try, not
    a guarantee: the router falls back to Tavily when it errors or finds
    nothing. `ddgs` does its own HTTP, so the rate-limit bucket is spent
    here explicitly rather than by the shared transport.
    """

    def __init__(
        self,
        settings: DuckDuckGoSettings,
        limiter: RateLimiter | None = None,
        backend: SearchBackend | None = None,
    ) -> None:
        self._settings = settings
        self._limiter = limiter
        self._backend = backend or self._ddgs

    @property
    def name(self) -> str:
        return _DUCKDUCKGO

    async def search(self, request: SearchRequest) -> SearchResponse:
        if self._limiter is not None:
            await self._limiter.acquire(self._settings.rate_bucket)
        vertical = _NEWS if request.topic == _NEWS else "text"
        rows = await asyncio.to_thread(self._backend, vertical, request)
        return SearchResponse(request=request, hits=_ddg_hits(rows), provider=_DUCKDUCKGO)

    def _ddgs(self, vertical: str, request: SearchRequest) -> list[dict[str, Any]]:
        from ddgs import DDGS

        client = DDGS(timeout=int(self._settings.timeout_seconds))
        search = client.news if vertical == _NEWS else client.text
        rows: list[dict[str, Any]] = search(
            request.query,
            region=self._settings.region,
            safesearch=self._settings.safesearch,
            timelimit=self._timelimit(request),
            max_results=request.max_results,
            backend=self._settings.backend,
        )
        return rows

    def _timelimit(self, request: SearchRequest) -> str | None:
        if request.recent_days is None:
            return None
        return window_name(request.recent_days, self._settings.time_limits)


def _ddg_hits(rows: list[dict[str, Any]]) -> list[SearchHit]:
    """`text` rows carry href/body; `news` rows carry url/body/date."""
    hits = []
    for row in rows:
        url = row.get("href") or row.get("url")
        if url:
            hits.append(
                SearchHit(
                    url=str(url),
                    title=str(row.get("title") or ""),
                    snippet=str(row.get("body") or ""),
                    published_date=str(row.get("date") or ""),
                )
            )
    return hits


def parse_tavily_response(request: SearchRequest, payload: dict[str, Any]) -> SearchResponse:
    """Map Tavily's JSON onto the provider-neutral response, tolerating absences."""
    hits = [
        SearchHit(
            url=str(item["url"]),
            title=str(item.get("title") or ""),
            snippet=str(item.get("content") or ""),
            score=float(item.get("score") or 0.0),
            published_date=str(item.get("published_date") or ""),
        )
        for item in payload.get("results") or []
        if isinstance(item, dict) and item.get("url")
    ]
    return SearchResponse(
        request=request,
        hits=hits,
        provider=_PROVIDER,
        provider_summary=str(payload.get("answer") or ""),
    )
