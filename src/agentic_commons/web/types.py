"""Web value types: a search request and its hits, a fetched page.

Two things are kept structurally apart because they have different
standing. A search *hit* is a pointer and a teaser written by someone
else's ranking system; a provider *summary* is prose a model wrote about
pages you have not read. Neither is evidence. Only a `FetchedPage` — the
text actually retrieved from a URL, with its hash — can support a claim.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from agentic_commons.hashing import canonical_hash
from agentic_commons.observability.types import utc_now

FetchVia = Literal["http", "reader"]


class SearchRequest(BaseModel):
    """One search, normalised so equal requests hash equally."""

    model_config = ConfigDict(frozen=True)

    query: str
    max_results: int
    include_domains: tuple[str, ...] = ()
    exclude_domains: tuple[str, ...] = ()

    def normalized(self) -> "SearchRequest":
        """Whitespace and case folded, domain lists sorted — the replay key's input."""
        return SearchRequest(
            query=" ".join(self.query.casefold().split()),
            max_results=self.max_results,
            include_domains=tuple(sorted(domain.casefold() for domain in self.include_domains)),
            exclude_domains=tuple(sorted(domain.casefold() for domain in self.exclude_domains)),
        )

    @property
    def key(self) -> str:
        return canonical_hash(self.normalized().model_dump(mode="json"))


class SearchHit(BaseModel):
    """A pointer to a page, with the provider's teaser (untrusted, not evidence)."""

    url: str
    title: str = ""
    snippet: str = ""
    score: float = 0.0
    published_date: str = ""


class SearchResponse(BaseModel):
    """What a provider returned for one request."""

    request: SearchRequest
    hits: list[SearchHit] = Field(default_factory=list)
    provider: str = ""
    # Provider-written prose (e.g. an "answer" field). Kept only so it can
    # be shown as untrusted; the gateway never turns it into a page.
    provider_summary: str = ""


class FetchedPage(BaseModel):
    """The text actually retrieved from one URL — the only thing that can be evidence."""

    url: str
    status: int
    text: str
    page_hash: str
    retrieved_at: datetime = Field(default_factory=utc_now)
    via: FetchVia = "http"
    title: str = ""

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300 and bool(self.text.strip())


def normalize_url(url: str) -> str:
    """The URL a cache and a replay fixture are keyed by."""
    stripped = url.strip()
    return stripped[:-1] if stripped.endswith("/") else stripped


def url_key(url: str) -> str:
    return canonical_hash({"url": normalize_url(url)})


class WebUnavailableError(RuntimeError):
    """A provider or site could not be reached after the resilience stack gave up."""


class ReplayMissError(LookupError):
    """Replay mode was asked for a request no fixture recorded. Never falls through."""
