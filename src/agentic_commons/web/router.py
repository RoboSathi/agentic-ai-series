"""Which search provider answers a request (Strategy + fallback chain).

Free first, metered when it matters: a general search goes to the first
provider in `web.search.providers` (DuckDuckGo) and falls through to the
next (Tavily) when a provider errors, is out of quota, or finds nothing.
A time-bounded or news search uses `time_bounded_providers` instead,
because that is where Tavily's time ranges and news vertical earn their
cost. Each provider has its own resilience stack, so one provider's
outage opens only its own breaker.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from agentic_commons.logging import get_logger
from agentic_commons.resilience import QuotaExhaustedError, ResilientCaller
from agentic_commons.web.providers import SearchProvider
from agentic_commons.web.types import SearchRequest, SearchResponse, WebUnavailableError

_ROUTER = "router"


@dataclass(frozen=True)
class RoutedProvider:
    """One provider and the resilience stack its calls go through."""

    provider: SearchProvider
    caller: ResilientCaller | None = None


class SearchRouter:
    """A `SearchProvider` that asks its providers in order until one has hits."""

    def __init__(
        self,
        providers: dict[str, RoutedProvider],
        general: Sequence[str],
        time_bounded: Sequence[str],
    ) -> None:
        self._providers = providers
        self._general = [name for name in general if name in providers]
        self._time_bounded = [name for name in time_bounded if name in providers]
        self._logger = get_logger("web")

    @property
    def name(self) -> str:
        return _ROUTER

    def order_for(self, request: SearchRequest) -> list[str]:
        return self._time_bounded if request.time_bounded else self._general

    async def search(self, request: SearchRequest) -> SearchResponse:
        failures: list[Exception] = []
        empty: SearchResponse | None = None
        for name in self.order_for(request):
            try:
                response = await self._ask(self._providers[name], request)
            except Exception as exc:
                failures.append(exc)
                self._logger.info("Search provider failed; trying the next", provider=name)
                continue
            if response.hits:
                return response
            empty = empty or response
        if empty is not None:
            return empty
        raise _exhausted(failures)

    @staticmethod
    async def _ask(routed: RoutedProvider, request: SearchRequest) -> SearchResponse:
        provider = routed.provider
        if routed.caller is None:
            return await provider.search(request)
        return await routed.caller.call(
            lambda: provider.search(request),
            idempotency_key=f"search:{provider.name}:{request.key}",
        )


def _exhausted(failures: list[Exception]) -> Exception:
    """Every provider out of quota is a quota stop; anything else is an outage."""
    reasons = "; ".join(str(exc) for exc in failures) or "no search provider is configured"
    if failures and all(isinstance(exc, QuotaExhaustedError) for exc in failures):
        return QuotaExhaustedError(reasons)
    return WebUnavailableError(reasons)
