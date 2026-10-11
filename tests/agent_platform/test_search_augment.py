"""A weak free answer also asks the metered provider, within a monthly allowance paced by day.

Nothing here opens a socket: the providers are scripted.
"""

from datetime import UTC, datetime

from agent_platform.config.settings import SearchAugmentSettings
from agent_platform.web import SearchRequest, SearchResponse
from agent_platform.web.augment import SearchAugmentPolicy
from agent_platform.web.providers import SearchHit
from agent_platform.web.router import RoutedProvider, SearchRouter

QUERY = "india medal tally asian games"
MID_OCTOBER = datetime(2026, 10, 15, 12, 0, tzinfo=UTC).timestamp()
SETTINGS = SearchAugmentSettings(
    provider="tavily", quota_bucket="tavily", min_hits=3, min_term_coverage=0.6,
    top_k=3, min_term_chars=3, share=0.5,
)  # fmt: skip


class _Provider:
    def __init__(self, name: str, hits: list[SearchHit]) -> None:
        self._name, self._hits = name, hits
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    async def search(self, request: SearchRequest) -> SearchResponse:
        self.calls += 1
        return SearchResponse(request=request, hits=self._hits, provider=self._name)


class _Quota:
    def __init__(self, used: int, quota: int | None = 900) -> None:
        self._used, self._quota = used, quota

    def quota_for(self, bucket: str) -> int | None:
        return self._quota

    def quota_used(self, bucket: str) -> int:
        return self._used


def _hit(url: str, title: str = QUERY) -> SearchHit:
    return SearchHit(url=url, title=title)


STRONG = [_hit(f"https://news.example/{i}") for i in range(5)]
OFF_TOPIC = [_hit(f"https://other.example/{i}", "cricket scores today") for i in range(5)]
TAVILY = [_hit("https://olympics.example/medals"), _hit("https://www.news.example/0/")]


def _router(ddg_hits: list[SearchHit], used: int = 0) -> tuple[SearchRouter, _Provider]:
    tavily = _Provider("tavily", TAVILY)
    policy = SearchAugmentPolicy(SETTINGS, _Quota(used), clock=lambda: MID_OCTOBER)
    router = SearchRouter(
        {"duckduckgo": RoutedProvider(_Provider("duckduckgo", ddg_hits)),
         "tavily": RoutedProvider(tavily)},
        ["duckduckgo", "tavily"], ["tavily", "duckduckgo"], policy,
    )  # fmt: skip
    return router, tavily


async def test_a_strong_free_answer_spends_nothing_metered() -> None:
    router, tavily = _router(STRONG)
    response = await router.search(SearchRequest(query=QUERY, max_results=8))
    assert response.provider == "duckduckgo" and tavily.calls == 0


async def test_too_few_hits_are_merged_with_the_metered_provider() -> None:
    router, tavily = _router(STRONG[:2])
    response = await router.search(SearchRequest(query=QUERY, max_results=8))
    assert tavily.calls == 1 and response.provider == "duckduckgo+tavily"
    # The free provider's best hit first, then the metered ones; a URL both found, once.
    assert [hit.url for hit in response.hits] == [
        "https://news.example/0",
        "https://olympics.example/medals",
        "https://news.example/1",
    ]


async def test_hits_that_barely_mention_the_query_are_weak() -> None:
    router, tavily = _router(OFF_TOPIC)
    await router.search(SearchRequest(query=QUERY, max_results=8))
    assert tavily.calls == 1


async def test_the_allowance_is_paced_over_the_month() -> None:
    # Mid-October, share 0.5 of 900: about 218 may be spent on augmenting by the 15th.
    router, tavily = _router(STRONG[:2], used=300)
    response = await router.search(SearchRequest(query=QUERY, max_results=8))
    assert tavily.calls == 0 and response.provider == "duckduckgo"


async def test_a_time_bounded_search_keeps_its_own_order() -> None:
    router, tavily = _router(STRONG[:1])
    response = await router.search(SearchRequest(query=QUERY, max_results=8, recent_days=7))
    assert response.provider == "tavily" and tavily.calls == 1


def test_no_quota_means_no_pacing() -> None:
    policy = SearchAugmentPolicy(SETTINGS, _Quota(10_000, quota=None))
    assert policy.allowed()


class _Down:
    name = "duckduckgo"

    async def search(self, request: SearchRequest) -> SearchResponse:
        raise RuntimeError("throttled")


async def test_the_route_says_who_answered_and_what_was_merged() -> None:
    router, _ = _router(STRONG)
    strong = await router.search(SearchRequest(query=QUERY, max_results=8))
    assert strong.route.answered_by == "duckduckgo" and not strong.route.augmented_with
    router, _ = _router(STRONG[:2])
    merged = await router.search(SearchRequest(query=QUERY, max_results=8))
    assert merged.route.answered_by == "duckduckgo" and merged.route.augmented_with == "tavily"


async def test_the_route_names_a_provider_that_failed_first() -> None:
    router = SearchRouter(
        {"duckduckgo": RoutedProvider(_Down()), "tavily": RoutedProvider(_Provider("tavily", TAVILY))},
        ["duckduckgo", "tavily"], ["tavily", "duckduckgo"],
    )  # fmt: skip
    response = await router.search(SearchRequest(query=QUERY, max_results=8))
    assert response.route.answered_by == "tavily" and response.route.failed == ["duckduckgo"]
