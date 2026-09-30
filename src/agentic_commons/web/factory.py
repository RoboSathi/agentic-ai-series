"""Build the configured web gateway (Factory).

The one place concrete network classes are chosen: the rate-limited HTTP
client (`llm.http`, which draws on the `tavily` bucket by host), one
resilience stack per dependency, and the fixture set named in config.
"""

from agentic_commons.config import get_settings
from agentic_commons.config.settings import WebMode, WebSettings
from agentic_commons.llm.http import create_http_client
from agentic_commons.resilience import ResilientCaller
from agentic_commons.web.fetcher import FallbackFetcher, HttpFetcher, ReaderFetcher
from agentic_commons.web.fixtures import FixtureStore
from agentic_commons.web.gateway import WebCallObserver, WebGateway
from agentic_commons.web.providers import TavilyProvider

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
        provider=TavilyProvider(resolved.tavily, client),
        fetcher=FallbackFetcher(HttpFetcher(resolved.fetch, client), reader, resolved.fetch),
        fixtures=fixtures if chosen == "record" else None,
        search_caller=ResilientCaller(_SEARCH_DEPENDENCY),
        fetch_caller=ResilientCaller(_FETCH_DEPENDENCY),
        observer=observer,
    )
