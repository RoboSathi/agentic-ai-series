"""The web gateway: one small interface over search and fetch, in three modes.

    live    call the provider and the site, through the resilience stack
    record  the same, and write every exchange to the fixture set
    replay  answer only from the fixture set; a miss raises, never falls through

Searches are memoised by request for the gateway's lifetime, and in live
mode an optional on-disk `SearchCache` reuses answers across runs, so a
repeated query spends no metered provider call. Only calls that actually
leave the machine are logged at INFO; replayed and cached ones at DEBUG.

Fetches are cached by URL for the gateway's lifetime, so fetching the
same page twice is free and returns identical text — which is what lets
the verbatim-span rule compare a span against *the* text of a page rather
than whichever copy the last request happened to return. The same cache
answers `text_for`, so the gateway is itself the evidence validator's
source of page text.
"""

from collections.abc import Awaitable, Callable
from typing import Literal, TypeVar

from pydantic import BaseModel

from agentic_commons.config.settings import WebMode
from agentic_commons.logging import get_logger
from agentic_commons.resilience import ResilientCaller
from agentic_commons.web.cache import SearchCache
from agentic_commons.web.fetcher import Fetcher
from agentic_commons.web.fixtures import FixtureStore
from agentic_commons.web.providers import SearchProvider
from agentic_commons.web.types import (
    FetchedPage,
    ReplayMissError,
    SearchRequest,
    SearchResponse,
    WebUnavailableError,
    normalize_url,
    url_key,
)

WebCallKind = Literal["search", "fetch"]
T = TypeVar("T")


class WebCall(BaseModel):
    """One gateway call, as an observer sees it."""

    kind: WebCallKind
    target: str
    mode: WebMode
    cached: bool = False
    ok: bool = True
    error: str = ""
    provider: str = ""

    @property
    def left_the_machine(self) -> bool:
        return self.mode != "replay" and not self.cached


WebCallObserver = Callable[[WebCall], None]


class WebGateway:
    """`search()` and `fetch()`, live, recorded or replayed."""

    def __init__(
        self,
        mode: WebMode,
        *,
        provider: SearchProvider | None = None,
        fetcher: Fetcher | None = None,
        fixtures: FixtureStore | None = None,
        search_caller: ResilientCaller | None = None,
        fetch_caller: ResilientCaller | None = None,
        observer: WebCallObserver | None = None,
        cache: SearchCache | None = None,
    ) -> None:
        if mode != "live" and fixtures is None:
            raise ValueError(f"web mode '{mode}' needs a fixture store")
        if mode != "replay" and (provider is None or fetcher is None):
            raise ValueError(f"web mode '{mode}' needs a provider and a fetcher")
        self._mode = mode
        self._provider = provider
        self._fetcher = fetcher
        self._fixtures = fixtures
        self._search_caller = search_caller
        self._fetch_caller = fetch_caller
        self._observer = observer
        self._cache = cache
        self._searches: dict[str, SearchResponse] = {}
        self._pages: dict[str, FetchedPage] = {}
        self._logger = get_logger("web")

    @property
    def mode(self) -> WebMode:
        return self._mode

    async def search(self, request: SearchRequest) -> SearchResponse:
        """Hits for one query; the provider's own summary is carried, flagged, never promoted."""
        remembered = self._remembered(request)
        if remembered is not None:
            self._notify(self._search_call(request, remembered, cached=True))
            return remembered
        try:
            response = await self._search(request)
        except Exception as exc:
            self._notify(
                WebCall(
                    kind="search", target=request.query, mode=self._mode, ok=False, error=str(exc)
                )
            )
            raise
        self._remember_search(response)
        self._notify(self._search_call(request, response, cached=False))
        return response

    def _remembered(self, request: SearchRequest) -> SearchResponse | None:
        memo = self._searches.get(request.key)
        if memo is None and self._mode == "live" and self._cache is not None:
            memo = self._cache.get(request)
        return memo

    def _remember_search(self, response: SearchResponse) -> None:
        self._searches[response.request.key] = response
        if self._mode == "live" and self._cache is not None:
            self._cache.put(response)

    def _search_call(
        self, request: SearchRequest, response: SearchResponse, cached: bool
    ) -> WebCall:
        return WebCall(
            kind="search",
            target=request.query,
            mode=self._mode,
            cached=cached,
            provider=response.provider,
        )

    async def fetch(self, url: str) -> FetchedPage:
        """A page's text, from the cache when this gateway already has it."""
        key = url_key(url)
        cached = self._pages.get(key)
        if cached is not None:
            self._notify(WebCall(kind="fetch", target=url, mode=self._mode, cached=True))
            return cached
        try:
            page = await self._fetch(url)
        except Exception as exc:
            self._notify(
                WebCall(kind="fetch", target=url, mode=self._mode, ok=False, error=str(exc))
            )
            raise
        self._pages[key] = page
        self._notify(WebCall(kind="fetch", target=url, mode=self._mode, ok=page.ok))
        return page

    def text_for(self, source_url: str) -> str | None:
        """The exact text this gateway fetched for a URL (the verbatim rule's source)."""
        page = self._pages.get(url_key(source_url))
        return page.text if page is not None else None

    def page_for(self, source_url: str) -> FetchedPage | None:
        return self._pages.get(url_key(source_url))

    def remember(self, page: FetchedPage) -> None:
        """Seed the cache with a page fetched earlier (a resumed run)."""
        self._pages[url_key(page.url)] = page

    async def _search(self, request: SearchRequest) -> SearchResponse:
        if self._mode == "replay":
            return self._replayed_search(request)
        response = await self._live_search(request)
        if self._mode == "record" and self._fixtures is not None:
            self._fixtures.put_search(request.key, response)
        return response

    async def _fetch(self, url: str) -> FetchedPage:
        if self._mode == "replay":
            return self._replayed_fetch(url)
        page = await self._live_fetch(url)
        if self._mode == "record" and self._fixtures is not None:
            self._fixtures.put_fetch(url_key(url), page)
        return page

    def _replayed_search(self, request: SearchRequest) -> SearchResponse:
        assert self._fixtures is not None
        # A fixture recorded without a time bound answers a bounded request:
        # the recorded web has no clock, and callers filter by date themselves.
        response = self._fixtures.search(request.key) or self._fixtures.search(
            request.unbounded().key
        )
        if response is None:
            raise ReplayMissError(
                f"no recorded search for '{request.normalized().query}' ({request.key[:12]})"
            )
        return response

    def _replayed_fetch(self, url: str) -> FetchedPage:
        assert self._fixtures is not None
        page = self._fixtures.fetch(url_key(url))
        if page is None:
            raise ReplayMissError(f"no recorded fetch for {normalize_url(url)}")
        return page

    async def _live_search(self, request: SearchRequest) -> SearchResponse:
        assert self._provider is not None
        provider = self._provider
        return await self._guarded(
            self._search_caller, lambda: provider.search(request), f"search:{request.key}"
        )

    async def _live_fetch(self, url: str) -> FetchedPage:
        assert self._fetcher is not None
        fetcher = self._fetcher
        return await self._guarded(
            self._fetch_caller, lambda: fetcher.fetch(url), f"fetch:{url_key(url)}"
        )

    async def _guarded(
        self, caller: ResilientCaller | None, call: Callable[[], Awaitable[T]], key: str
    ) -> T:
        """Through the resilience stack when one was given; an outage becomes one error type."""
        try:
            return await (caller.call(call, idempotency_key=key) if caller else call())
        except Exception as exc:
            # `str(exc)` is empty for timeouts and resets: name the exception instead.
            detail = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
            raise WebUnavailableError(f"{key.split(':')[0]} failed: {detail}") from exc

    def _notify(self, call: WebCall) -> None:
        log = self._logger.info if call.left_the_machine or not call.ok else self._logger.debug
        log(
            "Web call",
            provider=call.provider,
            kind=call.kind,
            target=call.target,
            mode=call.mode,
            cached=call.cached,
            ok=call.ok,
            error=call.error,
        )
        if self._observer is not None:
            self._observer(call)
