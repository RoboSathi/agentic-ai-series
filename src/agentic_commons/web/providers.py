"""Search providers (Strategy): who answers "which pages mention this?".

`TavilyProvider` is the v1 default. Its wire shape was taken from Tavily's
published API reference (`POST https://api.tavily.com/search`, bearer
auth; response `{query, answer, results[{title, url, content, score,
raw_content, published_date}], response_time, request_id}`); every field
but `results[].url` is treated as optional, because an adapter that
crashes on a missing teaser has made the teaser load-bearing.

`answer` is mapped to `provider_summary` and nothing else: it is text a
model wrote about pages nobody here has read, so it never becomes a page.
"""

from typing import Any, Protocol, runtime_checkable

import httpx

from agentic_commons.config.settings import TavilySettings
from agentic_commons.web.types import SearchHit, SearchRequest, SearchResponse

_PROVIDER = "tavily"


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
        return body


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
