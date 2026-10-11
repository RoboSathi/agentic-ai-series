"""Web access for agents: one gateway, live / record / replay (a generic mechanism).

Build it with `create_web_gateway()`; everything network-facing is
injected, so a test hands in a fixture store and never opens a socket.
"""

from agent_platform.web.factory import create_web_gateway
from agent_platform.web.fetcher import (
    FallbackFetcher,
    Fetcher,
    HttpFetcher,
    ReaderFetcher,
    extract_text,
    needs_reader,
    page_from,
)
from agent_platform.web.fixtures import FixtureManifest, FixtureStore
from agent_platform.web.gateway import WebCall, WebCallObserver, WebGateway
from agent_platform.web.providers import (
    MissingCredentialError,
    SearchProvider,
    TavilyProvider,
    parse_tavily_response,
)
from agent_platform.web.types import (
    FetchedPage,
    ReplayMissError,
    SearchHit,
    SearchRequest,
    SearchResponse,
    SearchRoute,
    WebUnavailableError,
    normalize_url,
    url_key,
)

__all__ = [
    "FallbackFetcher",
    "FetchedPage",
    "Fetcher",
    "FixtureManifest",
    "FixtureStore",
    "HttpFetcher",
    "MissingCredentialError",
    "ReaderFetcher",
    "ReplayMissError",
    "SearchHit",
    "SearchProvider",
    "SearchRequest",
    "SearchResponse",
    "SearchRoute",
    "TavilyProvider",
    "WebCall",
    "WebCallObserver",
    "WebGateway",
    "WebUnavailableError",
    "create_web_gateway",
    "extract_text",
    "needs_reader",
    "normalize_url",
    "page_from",
    "parse_tavily_response",
    "url_key",
]
