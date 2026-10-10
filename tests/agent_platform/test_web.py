"""The web gateway: live / record / replay, the Tavily adapter, fetching and the reader fallback.

Every "live" call here goes to an `httpx.MockTransport` — no socket is opened.
"""

import json
from pathlib import Path

import httpx
import pytest

from agent_platform.config import get_settings
from agent_platform.primitives.hashing import text_hash
from agent_platform.resilience import ResilientCaller
from agent_platform.resilience.circuit_breaker import CircuitBreakerImpl
from agent_platform.resilience.retry import ExponentialBackoffRetry
from agent_platform.web import (
    FallbackFetcher,
    FetchedPage,
    FixtureStore,
    HttpFetcher,
    MissingCredentialError,
    ReaderFetcher,
    ReplayMissError,
    SearchRequest,
    TavilyProvider,
    WebCall,
    WebGateway,
    WebUnavailableError,
    create_web_gateway,
    extract_text,
    needs_reader,
    parse_tavily_response,
    url_key,
)

SEARCH_URL = "https://api.tavily.com/search"
PAGE = "https://site.example/a"
SHELL = "https://site.example/app"
READER_PREFIX = "https://reader.test/"


def _handler(request: httpx.Request) -> httpx.Response:
    url = str(request.url)
    if url == SEARCH_URL:
        body = json.loads(request.content)
        assert request.headers["Authorization"] == "Bearer k"
        return httpx.Response(
            200,
            json={
                "query": body["query"],
                "answer": "A model wrote this.",
                "results": [
                    {
                        "url": PAGE,
                        "title": "A",
                        "content": "teaser",
                        "score": 0.9,
                        "published_date": "2026-01-01",
                    },
                    {"title": "no url"},
                ],
            },
        )
    if url == PAGE:
        return httpx.Response(
            200,
            text="<html><head><title>A</title><style>x{}</style></head><body><p>Alpha is 41, and this sentence is long enough to count as content.</p><script>bad()</script></body></html>",
        )
    if url == SHELL:
        return httpx.Response(
            200, text="<html><body><noscript>Please enable JavaScript.</noscript></body></html>"
        )
    if url.startswith(READER_PREFIX):
        return httpx.Response(
            200, text="Rendered content of the application page, long enough to count."
        )
    return httpx.Response(404, text="")


@pytest.fixture
def client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(_handler))


def _fetch_settings() -> object:
    return get_settings().web.fetch.model_copy(
        update={"reader_url_template": READER_PREFIX + "{url}", "min_text_chars": 20}
    )


def _live(
    client: httpx.AsyncClient,
    mode: str = "live",
    fixtures: FixtureStore | None = None,
    calls: list[WebCall] | None = None,
) -> WebGateway:
    fetch = _fetch_settings()
    return WebGateway(
        mode,  # type: ignore[arg-type]
        provider=TavilyProvider(get_settings().web.tavily, client, api_key="k"),
        fetcher=FallbackFetcher(HttpFetcher(fetch, client), ReaderFetcher(fetch, client), fetch),  # type: ignore[arg-type]
        fixtures=fixtures,
        observer=calls.append if calls is not None else None,
    )


class TestRequests:
    def test_equal_requests_share_a_key(self) -> None:
        a = SearchRequest(query="  Iron  AIR ", max_results=3, include_domains=("B.org", "a.org"))
        b = SearchRequest(query="iron air", max_results=3, include_domains=("a.org", "b.org"))
        assert a.key == b.key
        assert a.key != SearchRequest(query="iron air", max_results=4).key

    def test_url_keys_ignore_a_trailing_slash(self) -> None:
        assert url_key("https://x.example/") == url_key("https://x.example")


class TestTavily:
    def test_the_answer_is_a_summary_not_a_hit(self) -> None:
        response = parse_tavily_response(
            SearchRequest(query="q", max_results=2), {"answer": "prose", "results": [{"url": "u"}]}
        )
        assert response.provider_summary == "prose" and [h.url for h in response.hits] == ["u"]

    async def test_a_missing_key_refuses_to_call(
        self, client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("tavily_api_key", raising=False)
        provider = TavilyProvider(get_settings().web.tavily, client)
        with pytest.raises(MissingCredentialError):
            await provider.search(SearchRequest(query="q", max_results=1))

    async def test_domains_are_sent_when_given(self) -> None:
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return httpx.Response(200, json={"results": []})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            provider = TavilyProvider(get_settings().web.tavily, http, api_key="k")
            await provider.search(
                SearchRequest(
                    query="q", max_results=1, include_domains=("a.org",), exclude_domains=("b.org",)
                )
            )
        assert seen["include_domains"] == ["a.org"] and seen["exclude_domains"] == ["b.org"]
        assert seen["include_answer"] is False


class TestFetching:
    def test_extraction_drops_markup_scripts_and_styles(self) -> None:
        title, text = extract_text(
            "<title>T</title><style>s{}</style><p>One.</p><script>x()</script><p>Two.</p>"
        )
        assert title == "T" and "x()" not in text and "One." in text and "Two." in text

    def test_plain_text_passes_through(self) -> None:
        assert extract_text("just text") == ("", "just text")

    def test_a_script_shell_needs_the_reader(self) -> None:
        shell = FetchedPage(url="u", status=200, text="Please enable JavaScript.", page_hash="h")
        assert needs_reader(shell, _fetch_settings())  # type: ignore[arg-type]
        missing = shell.model_copy(update={"status": 404})
        assert not needs_reader(missing, _fetch_settings())  # type: ignore[arg-type]

    async def test_the_reader_is_used_for_a_shell_only(self, client: httpx.AsyncClient) -> None:
        gateway = _live(client)
        normal, shell = await gateway.fetch(PAGE), await gateway.fetch(SHELL)
        assert normal.via == "http" and "Alpha is 41" in normal.text and "bad()" not in normal.text
        assert shell.via == "reader" and "Rendered content" in shell.text
        assert normal.page_hash == text_hash(normal.text)


class TestModes:
    async def test_live_search_and_cached_fetch(self, client: httpx.AsyncClient) -> None:
        calls: list[WebCall] = []
        gateway = _live(client, calls=calls)
        response = await gateway.search(SearchRequest(query="alpha", max_results=2))
        assert response.provider_summary and [h.url for h in response.hits] == [PAGE]
        first = await gateway.fetch(PAGE)
        second = await gateway.fetch(PAGE)
        assert first is second and gateway.text_for(PAGE) == first.text
        assert [c.cached for c in calls if c.kind == "fetch"] == [False, True]

    async def test_record_then_replay_reproduces_the_run(
        self, client: httpx.AsyncClient, tmp_path: Path
    ) -> None:
        store = FixtureStore(tmp_path, "set1")
        recorder = _live(client, mode="record", fixtures=store)
        live = await recorder.search(SearchRequest(query="alpha", max_results=2))
        page = await recorder.fetch(PAGE)
        replay = WebGateway("replay", fixtures=store)
        assert (await replay.search(SearchRequest(query="ALPHA ", max_results=2))) == live
        assert (await replay.fetch(PAGE)).text == page.text
        assert store.keys("search") and store.keys("fetch")

    async def test_a_replay_miss_raises_and_never_falls_through(self, tmp_path: Path) -> None:
        replay = WebGateway("replay", fixtures=FixtureStore(tmp_path, "empty"))
        with pytest.raises(ReplayMissError):
            await replay.search(SearchRequest(query="anything", max_results=1))
        with pytest.raises(ReplayMissError):
            await replay.fetch("https://nowhere.example")

    async def test_an_outage_becomes_one_error_type(self, tmp_path: Path) -> None:
        def down(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="")

        async with httpx.AsyncClient(transport=httpx.MockTransport(down)) as http:
            caller = ResilientCaller(
                "t",
                breaker=CircuitBreakerImpl("t", failure_threshold=5),
                retry=ExponentialBackoffRetry(max_attempts=1),
            )
            gateway = WebGateway(
                "live",
                provider=TavilyProvider(get_settings().web.tavily, http, api_key="k"),
                fetcher=HttpFetcher(get_settings().web.fetch, http),
                search_caller=caller,
            )
            with pytest.raises(WebUnavailableError):
                await gateway.search(SearchRequest(query="q", max_results=1))

    def test_modes_demand_their_collaborators(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            WebGateway("replay")
        with pytest.raises(ValueError):
            WebGateway("live")

    def test_the_factory_builds_a_replaying_gateway_by_default(self) -> None:
        gateway = create_web_gateway()
        assert gateway.mode == "replay"

    def test_the_factory_builds_a_live_stack_when_asked(self) -> None:
        assert create_web_gateway(mode="live").mode == "live"

    def test_remember_seeds_the_cache(self) -> None:
        gateway = create_web_gateway()
        gateway.remember(FetchedPage(url=PAGE, status=200, text="t", page_hash=text_hash("t")))
        assert gateway.page_for(PAGE) is not None and gateway.text_for(PAGE) == "t"
