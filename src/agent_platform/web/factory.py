"""Build the configured web gateway (Factory).

The one place concrete network classes are chosen: the rate-limited HTTP
client (`llm.http`, which draws on the `tavily` bucket by host), the search
router over the configured providers (each with its own resilience stack),
the on-disk search cache, and the fixture set named in config.
"""

import httpx

from agent_platform.config import get_settings
from agent_platform.config.settings import WebMode, WebSettings
from agent_platform.llm.http import create_http_client
from agent_platform.resilience import ExponentialBackoffRetry, ResilientCaller
from agent_platform.resilience.rate_limiter import get_rate_limiter
from agent_platform.web.cache import SearchCache
from agent_platform.web.fetcher import FallbackFetcher, HttpFetcher, ReaderFetcher
from agent_platform.web.fixtures import FixtureStore
from agent_platform.web.gateway import WebCallObserver, WebGateway
from agent_platform.web.providers import DuckDuckGoProvider, SearchProvider, TavilyProvider
from agent_platform.web.router import RoutedProvider, SearchRouter

_SEARCH_DEPENDENCY = "web_search"
_FETCH_DEPENDENCY = "web_fetch"


def create_web_gateway(
    settings: WebSettings | None = None,
    *,
    mode: WebMode | None = None,
    fixture_set: str | None = None,
    observer: WebCallObserver | None = None,
) -> WebGateway:
    """The gateway configuration describes; `mode`/`fixture_set` override it."""
    resolved = settings or get_settings().web
    chosen = mode or resolved.mode
    fixtures = FixtureStore(resolved.resolved_fixtures_dir, fixture_set or resolved.fixture_set)
    if chosen == "replay":
        return WebGateway("replay", fixtures=fixtures, observer=observer)
    client = create_http_client(resolved.fetch.timeout_seconds)
    reader = ReaderFetcher(resolved.fetch, client) if resolved.fetch.reader_enabled else None
    return WebGateway(
        chosen,
        provider=create_search_router(resolved, client),
        fetcher=FallbackFetcher(HttpFetcher(resolved.fetch, client), reader, resolved.fetch),
        fixtures=fixtures if chosen == "record" else None,
        fetch_caller=ResilientCaller(
            _FETCH_DEPENDENCY,
            retry=ExponentialBackoffRetry(max_attempts=resolved.fetch.max_attempts),
        ),
        observer=observer,
        cache=SearchCache(resolved.search.resolved_cache_dir, resolved.search.cache_ttl_hours),
    )


def create_search_router(settings: WebSettings, client: httpx.AsyncClient) -> SearchRouter:
    """Every configured provider behind one router, each with its own breaker."""
    built: dict[str, SearchProvider] = {
        "tavily": TavilyProvider(settings.tavily, client),
        "duckduckgo": DuckDuckGoProvider(settings.duckduckgo, get_rate_limiter()),
    }
    return SearchRouter(
        {
            name: RoutedProvider(provider, ResilientCaller(f"{_SEARCH_DEPENDENCY}_{name}"))
            for name, provider in built.items()
        },
        settings.search.providers,
        settings.search.time_bounded_providers,
    )
