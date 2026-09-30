"""Monthly quotas, the search router, DuckDuckGo, Tavily time ranges and the search cache.

Nothing here opens a socket: Tavily sits behind an `httpx.MockTransport`,
DuckDuckGo behind an injected backend function.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from agentic_commons.config import get_settings
from agentic_commons.config.settings import RateLimitBucketSettings, RateLimitSettings
from agentic_commons.resilience import QuotaExhaustedError, ResilientCaller
from agentic_commons.resilience.rate_limiter import (
    InMemoryRateWindowStore,
    SlidingWindowRateLimiter,
    SqliteRateWindowStore,
)
from agentic_commons.resilience.retry import ExponentialBackoffRetry
from agentic_commons.web import (
    FallbackFetcher,
    HttpFetcher,
    SearchRequest,
    SearchResponse,
    TavilyProvider,
    WebCall,
    WebGateway,
    WebUnavailableError,
)
from agentic_commons.web.cache import SearchCache
from agentic_commons.web.providers import DuckDuckGoProvider, SearchHit, window_name
from agentic_commons.web.router import RoutedProvider, SearchRouter

SEPTEMBER = datetime(2026, 9, 30, 23, 0, tzinfo=UTC).timestamp()
OCTOBER = datetime(2026, 10, 1, 1, 0, tzinfo=UTC).timestamp()


def _settings(quota: int | None) -> RateLimitSettings:
    return RateLimitSettings(
        enabled=True,
        store_path=":unused:",
        window_seconds=60.0,
        max_wait_seconds=600.0,
        lock_timeout_seconds=5.0,
        default_retry_after_seconds=30.0,
        buckets={
            "metered": RateLimitBucketSettings(max_requests=1000, hosts=[], monthly_quota=quota)
        },
    )


class Clock:
    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class TestMonthlyQuota:
    async def test_the_request_past_the_quota_fails_fast(self) -> None:
        limiter = SlidingWindowRateLimiter(
            _settings(3), InMemoryRateWindowStore(), clock=Clock(SEPTEMBER)
        )
        for _ in range(3):
            await limiter.acquire("metered")
        with pytest.raises(QuotaExhaustedError):
            await limiter.acquire("metered")
        assert limiter.quota_used("metered") == 3

    async def test_the_quota_resets_with_the_month(self) -> None:
        clock = Clock(SEPTEMBER)
        limiter = SlidingWindowRateLimiter(_settings(1), InMemoryRateWindowStore(), clock=clock)
        await limiter.acquire("metered")
        clock.now = OCTOBER
        await limiter.acquire("metered")
        assert limiter.quota_used("metered") == 1

    async def test_every_process_shares_one_count(self, tmp_path: Path) -> None:
        path = tmp_path / "rate.sqlite"
        first = SlidingWindowRateLimiter(
            _settings(2), SqliteRateWindowStore(path, 5.0), clock=Clock(SEPTEMBER)
        )
        second = SlidingWindowRateLimiter(
            _settings(2), SqliteRateWindowStore(path, 5.0), clock=Clock(SEPTEMBER)
        )
        await first.acquire("metered")
        await second.acquire("metered")
        with pytest.raises(QuotaExhaustedError):
            await first.acquire("metered")
        assert second.quota_used("metered") == 2

    async def test_no_quota_means_no_cap(self) -> None:
        limiter = SlidingWindowRateLimiter(
            _settings(None), InMemoryRateWindowStore(), clock=Clock(SEPTEMBER)
        )
        await limiter.acquire("metered")
        assert limiter.quota_for("metered") is None

    def test_a_spent_quota_is_never_retried(self) -> None:
        assert not ExponentialBackoffRetry().should_retry(QuotaExhaustedError("spent"))

    def test_tavily_is_capped_under_its_free_tier(self) -> None:
        tavily = get_settings().resilience.rate_limit.buckets["tavily"]
        assert tavily.monthly_quota is not None and tavily.monthly_quota < 1000


class TestRequestKeys:
    def test_old_requests_keep_their_replay_key(self) -> None:
        # Recorded before `recent_days`/`topic` existed: the key must not move.
        from agentic_commons.hashing import canonical_hash

        plain = SearchRequest(query="iron air", max_results=3)
        legacy = {
            "query": "iron air",
            "max_results": 3,
            "include_domains": [],
            "exclude_domains": [],
        }
        assert plain.key == canonical_hash(legacy)

    def test_time_bounds_change_the_key(self) -> None:
        plain = SearchRequest(query="ai news", max_results=3)
        assert SearchRequest(query="ai news", max_results=3, recent_days=30).key != plain.key
        assert SearchRequest(query="ai news", max_results=3, topic="news").key != plain.key


class TestWindows:
    @pytest.mark.parametrize(
        ("days", "name"), [(1, "day"), (5, "week"), (30, "month"), (90, "year"), (999, "year")]
    )
    def test_the_narrowest_covering_window(self, days: int, name: str) -> None:
        assert window_name(days, get_settings().web.tavily.time_ranges) == name

    async def test_tavily_sends_the_time_range_and_topic(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return httpx.Response(200, json={"results": []})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            provider = TavilyProvider(get_settings().web.tavily, http, api_key="k")
            await provider.search(
                SearchRequest(query="ai", max_results=1, recent_days=30, topic="news")
            )
        assert seen["time_range"] == "month" and seen["topic"] == "news"


class TestDuckDuckGo:
    async def test_text_and_news_rows_become_hits(self) -> None:
        asked: list[tuple[str, SearchRequest]] = []

        def backend(vertical: str, request: SearchRequest) -> list[dict[str, Any]]:
            asked.append((vertical, request))
            if vertical == "news":
                return [
                    {"url": "https://n.example/1", "title": "N", "body": "b", "date": "2026-09-01"}
                ]
            return [{"href": "https://t.example/1", "title": "T", "body": "b"}, {"title": "no url"}]

        provider = DuckDuckGoProvider(get_settings().web.duckduckgo, backend=backend)
        text = await provider.search(SearchRequest(query="q", max_results=3))
        news = await provider.search(SearchRequest(query="q", max_results=3, topic="news"))
        assert [h.url for h in text.hits] == ["https://t.example/1"]
        assert news.hits[0].published_date == "2026-09-01" and news.provider == "duckduckgo"
        assert [vertical for vertical, _ in asked] == ["text", "news"]

    async def test_it_spends_its_own_rate_bucket(self) -> None:
        settings = get_settings().web.duckduckgo
        limits = _settings(None).model_copy(
            update={
                "buckets": {settings.rate_bucket: RateLimitBucketSettings(max_requests=5, hosts=[])}
            }
        )
        limiter = SlidingWindowRateLimiter(limits, InMemoryRateWindowStore())
        provider = DuckDuckGoProvider(settings, limiter, backend=lambda v, r: [])
        await provider.search(SearchRequest(query="q", max_results=1))
        assert limiter.stats.bucket(settings.rate_bucket).requests == 1


class Scripted:
    """A provider that answers from a script and counts its calls."""

    def __init__(self, name: str, hits: int = 1, error: Exception | None = None) -> None:
        self._name, self._hits, self._error = name, hits, error
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    async def search(self, request: SearchRequest) -> SearchResponse:
        self.calls += 1
        if self._error is not None:
            raise self._error
        hits = [SearchHit(url=f"https://{self._name}.example/{i}") for i in range(self._hits)]
        return SearchResponse(request=request, hits=hits, provider=self._name)


def _router(ddg: Scripted, tavily: Scripted) -> SearchRouter:
    return SearchRouter(
        {"duckduckgo": RoutedProvider(ddg), "tavily": RoutedProvider(tavily)},
        ["duckduckgo", "tavily"],
        ["tavily", "duckduckgo"],
    )


class TestRouter:
    async def test_general_searches_try_the_free_provider_first(self) -> None:
        ddg, tavily = Scripted("duckduckgo"), Scripted("tavily")
        response = await _router(ddg, tavily).search(SearchRequest(query="q", max_results=2))
        assert response.provider == "duckduckgo" and tavily.calls == 0

    async def test_time_bounded_searches_prefer_tavily(self) -> None:
        ddg, tavily = Scripted("duckduckgo"), Scripted("tavily")
        request = SearchRequest(query="q", max_results=2, recent_days=30)
        assert (await _router(ddg, tavily).search(request)).provider == "tavily"
        assert ddg.calls == 0

    async def test_empty_or_failing_providers_fall_through(self) -> None:
        ddg, tavily = Scripted("duckduckgo", hits=0), Scripted("tavily")
        assert (
            await _router(ddg, tavily).search(SearchRequest(query="q", max_results=2))
        ).provider == "tavily"
        spent = Scripted("tavily", error=QuotaExhaustedError("spent"))
        request = SearchRequest(query="q", max_results=2, topic="news")
        assert (
            await _router(Scripted("duckduckgo"), spent).search(request)
        ).provider == "duckduckgo"

    async def test_nothing_answers_raises_one_error(self) -> None:
        down = RuntimeError("down")
        router = _router(Scripted("duckduckgo", error=down), Scripted("tavily", error=down))
        with pytest.raises(WebUnavailableError):
            await router.search(SearchRequest(query="q", max_results=2))

    async def test_a_provider_goes_through_its_own_resilience_stack(self) -> None:
        tavily = Scripted("tavily")
        router = SearchRouter(
            {"tavily": RoutedProvider(tavily, ResilientCaller("t"))}, ["tavily"], ["tavily"]
        )
        await router.search(SearchRequest(query="q", max_results=1))
        assert tavily.calls == 1


class TestSearchCache:
    def _gateway(self, provider: Scripted, cache: SearchCache, calls: list[WebCall]) -> WebGateway:
        fetch = get_settings().web.fetch
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(404)))
        return WebGateway(
            "live",
            provider=provider,
            fetcher=FallbackFetcher(HttpFetcher(fetch, client), None, fetch),
            cache=cache,
            observer=calls.append,
        )

    async def test_a_cached_search_costs_no_provider_call(self, tmp_path: Path) -> None:
        provider, calls = Scripted("tavily"), []
        cache = SearchCache(tmp_path, ttl_hours=24)
        request = SearchRequest(query="q", max_results=1)
        await self._gateway(provider, cache, calls).search(request)
        await self._gateway(provider, cache, calls).search(request)  # a later run
        assert provider.calls == 1 and [c.cached for c in calls] == [False, True]

    async def test_a_repeat_within_one_run_is_memoised(self, tmp_path: Path) -> None:
        provider, calls = Scripted("tavily"), []
        gateway = self._gateway(provider, SearchCache(tmp_path, ttl_hours=0), calls)
        request = SearchRequest(query="q", max_results=1)
        await gateway.search(request)
        await gateway.search(request)
        assert provider.calls == 1

    def test_entries_expire(self, tmp_path: Path) -> None:
        clock = Clock(SEPTEMBER)
        cache = SearchCache(tmp_path, ttl_hours=1, clock=clock)
        request = SearchRequest(query="q", max_results=1)
        cache.put(SearchResponse(request=request, provider="tavily"))
        assert cache.get(request) is not None
        clock.now += 2 * 3600
        assert cache.get(request) is None

    def test_a_replayed_call_did_not_leave_the_machine(self) -> None:
        assert not WebCall(kind="search", target="q", mode="replay").left_the_machine
        assert WebCall(kind="search", target="q", mode="live").left_the_machine


class TestReplayOfBoundedRequests:
    async def test_an_unbounded_recording_answers_a_bounded_request(self, tmp_path: Path) -> None:
        from agentic_commons.web import FixtureStore

        fixtures = FixtureStore(tmp_path, "set")
        plain = SearchRequest(query="ai news", max_results=2)
        fixtures.put_search(plain.key, SearchResponse(request=plain, provider="tavily"))
        gateway = WebGateway("replay", fixtures=fixtures)
        bounded = SearchRequest(query="ai news", max_results=2, recent_days=30, topic="news")
        assert (await gateway.search(bounded)).provider == "tavily"
