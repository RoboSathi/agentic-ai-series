"""Web access for agents: one gateway, live / record / replay (a generic mechanism).

Build it with `create_web_gateway()`; everything network-facing is
injected, so a test hands in a fixture store and never opens a socket.
"""

from agentic_commons.web.factory import create_web_gateway
from agentic_commons.web.fetcher import (
    FallbackFetcher,
    Fetcher,
    HttpFetcher,
    ReaderFetcher,
    extract_text,
    needs_reader,
    page_from,
)
from agentic_commons.web.fixtures import FixtureManifest, FixtureStore
from agentic_commons.web.gateway import WebCall, WebCallObserver, WebGateway
from agentic_commons.web.providers import (
    MissingCredentialError,
    SearchProvider,
    TavilyProvider,
    parse_tavily_response,
)
from agentic_commons.web.types import (
    FetchedPage,
    ReplayMissError,
    SearchHit,
    SearchRequest,
    SearchResponse,
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
