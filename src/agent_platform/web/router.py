"""Which search provider answers a request (Strategy + fallback chain).

Free first, metered when it matters: a general search goes to the first
provider in `web.search.providers` (DuckDuckGo) and falls through to the
next (Tavily) when a provider errors, is out of quota, or finds nothing.
A time-bounded or news search uses `time_bounded_providers` instead,
because that is where Tavily's time ranges and news vertical earn their
cost. Each provider has its own resilience stack, so one provider's
outage opens only its own breaker.

A general search the free provider answers *weakly* (few hits, or hits
that barely mention the query) also asks the metered one and merges the
two, within a monthly allowance (`augment.py`). A strong answer costs
nothing metered.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from agent_platform.logging import get_logger
from agent_platform.resilience import QuotaExhaustedError, ResilientCaller
from agent_platform.web.augment import SearchAugmentPolicy
from agent_platform.web.providers import SearchProvider
from agent_platform.web.types import (
    SearchRequest,
    SearchResponse,
    SearchRoute,
    WebUnavailableError,
)

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
        augment: SearchAugmentPolicy | None = None,
    ) -> None:
        self._providers = providers
        self._augment = augment
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
        failed: list[str] = []
        empty: SearchResponse | None = None
        for name in self.order_for(request):
            try:
                response = await self._ask(self._providers[name], request)
            except Exception as exc:
                failures.append(exc)
                failed.append(name)
                self._logger.info("Search provider failed; trying the next", provider=name)
                continue
            if response.hits:
                merged = await self._augmented(name, request, response)
                extra = self._augment.provider if merged is not response and self._augment else ""
                return _routed(merged, name, failed, extra)
            empty = empty or _routed(response, name, failed)
        if empty is not None:
            return empty
        raise _exhausted(failures)

    async def _augmented(
        self, answered_by: str, request: SearchRequest, response: SearchResponse
    ) -> SearchResponse:
        """A weak general answer, merged with the metered provider's when the allowance permits."""
        policy = self._augment
        if (
            policy is None
            or request.time_bounded
            or answered_by == policy.provider
            or policy.provider not in self._providers
        ):
            return response
        measured = policy.weakness(response)
        if not measured.weak:
            policy.log(request, measured, "strong")
            return response
        if not policy.allowed():
            policy.log(request, measured, "allowance spent")
            return response
        try:
            extra = await self._ask(self._providers[policy.provider], request)
        except Exception as exc:
            self._logger.info("Augmenting search failed", provider=policy.provider, error=str(exc))
            return response
        policy.log(request, measured, f"merged with {policy.provider}")
        return policy.merge(response, extra)

    @staticmethod
    async def _ask(routed: RoutedProvider, request: SearchRequest) -> SearchResponse:
        provider = routed.provider
        if routed.caller is None:
            return await provider.search(request)
        return await routed.caller.call(
            lambda: provider.search(request),
            idempotency_key=f"search:{provider.name}:{request.key}",
        )


def _routed(
    response: SearchResponse, answered_by: str, failed: list[str], augmented_with: str = ""
) -> SearchResponse:
    """The response, carrying how it was reached."""
    route = SearchRoute(answered_by=answered_by, failed=list(failed), augmented_with=augmented_with)
    return response.model_copy(update={"route": route})


def _exhausted(failures: list[Exception]) -> Exception:
    """Every provider out of quota is a quota stop; anything else is an outage."""
    reasons = "; ".join(str(exc) for exc in failures) or "no search provider is configured"
    if failures and all(isinstance(exc, QuotaExhaustedError) for exc in failures):
        return QuotaExhaustedError(reasons)
    return WebUnavailableError(reasons)
