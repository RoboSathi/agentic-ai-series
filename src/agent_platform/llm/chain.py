"""Composing one protected model client: the only place it is done (Builder).

A call is protected in two layers, and both are platform code:

    SteppedClient                                one logical call = one step of the turn
      └ CascadingLLMClient(hedged, label)          which model answers: primary, then backups
          └ ResilientLLMClient(model, ResilientCaller)  each model: bulkhead → breaker → retry → timeout
              └ RequestExtras(adapter, extra_body)      the model's own request fields
                  └ RecordedClient                      each attempt = one `fact_model_call` row

Roles (`RoleModelRouter`) and the default client (`create_llm_client`) both build through
this, from `ModelRef` + `ModelChainSettings`, so no agent composes resilience itself.
"""

from collections.abc import Callable, Mapping
from typing import Any

from agent_platform.config import get_settings
from agent_platform.config.settings import LLMSettings, ModelChainSettings, ModelRef, ProviderName
from agent_platform.llm.call_log import RecordedClient, SteppedClient
from agent_platform.llm.cascade import CascadingLLMClient
from agent_platform.llm.client import LLMClient
from agent_platform.llm.observer import ModelCallObserver
from agent_platform.llm.resilient import ResilientLLMClient
from agent_platform.llm.types import CompletionRequest, CompletionResult, HealthCheckResult
from agent_platform.resilience import ResilientCaller

ClientFactory = Callable[[ProviderName, str], LLMClient]
# A model's resilience stack is named after the model: `llm:<provider>:<model>`.
DEPENDENCY_PREFIX = "llm:"


class RequestExtras(LLMClient):
    """A client whose every request carries a model's provider-specific fields (Decorator).

    Two roles on one model can differ in how they ask — the interviewer with reasoning off,
    the answerer with it on — without a second adapter or a second connection pool.
    """

    def __init__(self, client: LLMClient, extra_body: Mapping[str, Any]) -> None:
        self._client = client
        self._extra = dict(extra_body)

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        extra = {**self._extra, **request.extra_body}
        return await self._client.complete(request.model_copy(update={"extra_body": extra}))

    async def health_check(self) -> HealthCheckResult:
        return await self._client.health_check()

    @property
    def model(self) -> str:
        return self._client.model

    @property
    def tier(self) -> str:
        return self._client.tier


class ModelChainBuilder:
    """Builds protected clients; one adapter and one resilience stack per distinct model."""

    def __init__(
        self,
        factory: ClientFactory,
        observer: ModelCallObserver | None = None,
        settings: LLMSettings | None = None,
    ) -> None:
        self._factory = factory
        self._observer = observer
        self._settings = settings or get_settings().llm
        self._adapters: dict[str, LLMClient] = {}
        self._guards: dict[str, ResilientCaller] = {}

    def build(self, primary: ModelRef, chain: ModelChainSettings, label: str) -> LLMClient:
        """The primary model, protected, then each backup in order (hedged when configured)."""
        refs = [primary, *chain.fallbacks]
        tiers = [
            self._protected(ref, chain.timeout_seconds, label, position)
            for position, ref in enumerate(refs)
        ]
        if len(tiers) == 1:
            return SteppedClient(tiers[0])
        # Backups are about availability: a weak answer is the role's to judge, not the chain's.
        availability = self._settings.cascade.model_copy(update={"escalate_on_finish_reasons": []})
        return SteppedClient(
            CascadingLLMClient(
                tiers,
                availability,
                hedge_after_seconds=chain.hedge_after_seconds,
                observer=self._observer,
                label=label,
            )
        )

    def _protected(
        self, ref: ModelRef, timeout_seconds: float | None, label: str, position: int
    ) -> LLMClient:
        if ref.key not in self._adapters:
            self._adapters[ref.key] = self._factory(ref.provider, ref.model)
        client: LLMClient = RecordedClient(self._adapters[ref.key], role=label, position=position)
        if ref.extra_body:
            client = RequestExtras(client, ref.extra_body)
        guard = self._guards.setdefault(ref.key, ResilientCaller(f"{DEPENDENCY_PREFIX}{ref.key}"))
        return ResilientLLMClient(client, guard, timeout_seconds)


__all__ = ["ClientFactory", "ModelChainBuilder", "RequestExtras"]
