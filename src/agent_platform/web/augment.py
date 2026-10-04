"""When a free search looks weak, ask the metered one too — within a monthly allowance.

DuckDuckGo answers a general search for free; Tavily is metered (a monthly quota). Falling through
to Tavily only when DuckDuckGo *errors or finds nothing* leaves the common bad case alone: a
handful of hits, or hits that barely mention what was asked. This policy names that case and
paces how often it may spend the metered provider on it:

    weak       fewer than `min_hits` hits, or the query's content words barely appear in the top
               `top_k` titles and snippets (a lexical check: no model, no network)
    allowance  `quota × (day of month / days in month) × share` — spent evenly across the month,
               the rest (`1 − share`) kept for pure fallback and for research runs

The merge keeps the free provider's order, puts the metered hits after its best one, drops a URL
both returned, and caps the list at the request's `max_results`.
"""

import calendar
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import NamedTuple, Protocol

from agent_platform.config.settings import SearchAugmentSettings
from agent_platform.llm.embeddings import tokenize
from agent_platform.logging import get_logger
from agent_platform.web.types import SearchHit, SearchRequest, SearchResponse

Clock = Callable[[], float]


class QuotaReader(Protocol):
    """What the allowance needs of the rate limiter: a bucket's quota and this month's spend."""

    def quota_for(self, bucket: str) -> int | None: ...

    def quota_used(self, bucket: str) -> int: ...


class Weakness(NamedTuple):
    """How a response measured against the bars, and whether it fell below one."""

    weak: bool
    hits: int
    coverage: float


class SearchAugmentPolicy:
    """Decides whether a response is weak, whether the allowance permits, and merges."""

    def __init__(
        self,
        settings: SearchAugmentSettings,
        quota: QuotaReader | None,
        clock: Clock = time.time,
    ) -> None:
        self._settings = settings
        self._quota = quota
        self._clock = clock
        self._logger = get_logger("web")

    @property
    def provider(self) -> str:
        """The metered provider asked when a response is weak."""
        return self._settings.provider

    def weakness(self, response: SearchResponse) -> Weakness:
        terms = self._content_words(response.request.query)
        top = response.hits[: self._settings.top_k]
        seen = set(tokenize(" ".join(f"{hit.title} {hit.snippet}" for hit in top)))
        coverage = len(terms & seen) / len(terms) if terms else 1.0
        weak = len(response.hits) < self._settings.min_hits or (
            coverage < self._settings.min_term_coverage
        )
        return Weakness(weak=weak, hits=len(response.hits), coverage=coverage)

    def allowed(self) -> bool:
        """Whether this month's paced allowance has room for one more metered search."""
        if self._quota is None:
            return True
        quota = self._quota.quota_for(self._settings.quota_bucket)
        if quota is None:
            return True
        now = datetime.fromtimestamp(self._clock(), tz=UTC)
        days = calendar.monthrange(now.year, now.month)[1]
        allowance = quota * (now.day / days) * self._settings.share
        return self._quota.quota_used(self._settings.quota_bucket) < allowance

    def merge(self, primary: SearchResponse, extra: SearchResponse) -> SearchResponse:
        """The free provider's best hit, the metered hits, then the rest — no URL twice."""
        ordered = [*primary.hits[:1], *extra.hits, *primary.hits[1:]]
        hits: list[SearchHit] = []
        seen: set[str] = set()
        for hit in ordered:
            key = _normalised(hit.url)
            if key not in seen:
                seen.add(key)
                hits.append(hit)
        provider = f"{primary.provider}+{extra.provider}"
        return primary.model_copy(
            update={"hits": hits[: primary.request.max_results], "provider": provider}
        )

    def log(self, request: SearchRequest, measured: Weakness, outcome: str) -> None:
        self._logger.info(
            "Search quality check",
            query=request.query, hits=measured.hits, coverage=round(measured.coverage, 2),
            min_hits=self._settings.min_hits, min_coverage=self._settings.min_term_coverage,
            weak=measured.weak, outcome=outcome,
        )  # fmt: skip

    def _content_words(self, query: str) -> set[str]:
        return {word for word in tokenize(query) if len(word) >= self._settings.min_term_chars}


def _normalised(url: str) -> str:
    """A URL without scheme, `www.` or a trailing slash: the same page from two providers."""
    bare = url.split("://", 1)[-1].removeprefix("www.")
    return bare.rstrip("/").casefold()


__all__ = ["QuotaReader", "SearchAugmentPolicy", "Weakness"]
