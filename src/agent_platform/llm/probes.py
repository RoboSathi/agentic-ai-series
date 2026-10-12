"""Endpoint probes for `make model-check` — one per hosted model the system calls.

Each probe sends the smallest *real* request its adapter can make,
through the same adapter the agent uses (and so through the rate
limiter). Listing models is not proof: NIM entitlements are per account,
so a model can be listed and still refuse to answer.

Which models to probe is the inventory's job (`llm.inventory`): every model every
configuration layer names, once; a section switched off names no model.
"""

import asyncio
import time
from typing import Awaitable, Callable, Literal, Protocol, runtime_checkable

from pydantic import BaseModel

from agent_platform.config import AppConfig, get_settings
from agent_platform.config.settings import ProviderName, SecurityGuardModelSettings
from agent_platform.llm.client import create_pinned_llm_client
from agent_platform.llm.embeddings import HostedEmbeddingClient
from agent_platform.llm.types import CompletionRequest, Message
from agent_platform.resilience import ResilientCaller, admission
from agent_platform.security.content_moderation.hosted import HostedContentSafetyGuard
from agent_platform.security.guard_types import GuardVerdict
from agent_platform.security.prompt_injection.hosted import HostedJailbreakGuard

ProbeStatus = Literal["ok", "failed", "skipped"]


class ProbeResult(BaseModel):
    """What one endpoint did when asked the smallest real question."""

    name: str
    target: str
    model: str
    status: ProbeStatus
    required: bool
    latency_ms: float = 0.0
    http_status: int | None = None
    detail: str = ""


@runtime_checkable
class EndpointProbe(Protocol):
    """Sends one minimal request to one hosted endpoint."""

    name: str

    async def probe(self) -> ProbeResult: ...


class _TimedProbe:
    """Shared bookkeeping: time the call, turn any exception into `failed`."""

    def __init__(self, name: str, target: str, model: str, required: bool) -> None:
        self.name = name
        self._target = target
        self._model = model
        self._required = required
        self._timeout = get_settings().llm.check.timeout_seconds

    def result(self, status: ProbeStatus, started: float, **fields: object) -> ProbeResult:
        return ProbeResult(
            name=self.name,
            target=self._target,
            model=self._model,
            status=status,
            required=self._required,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            **fields,  # type: ignore[arg-type]
        )

    async def timed(self, call: Callable[[], Awaitable[ProbeResult]]) -> ProbeResult:
        started = time.perf_counter()
        try:
            # The deadline covers the endpoint, not the rate-limit queue: a probe queued behind a
            # running eval for its slot is waiting, not failing (`resilience.admission`).
            async with asyncio.timeout(self._timeout) as scope:
                with admission.deadline_scope(scope):
                    return await call()
        except TimeoutError:
            return self.result("failed", started, detail=f"no response within {self._timeout:g}s")
        except Exception as exc:
            return self.result("failed", started, http_status=_status_of(exc), detail=str(exc))


class ChatProbe(_TimedProbe):
    """A one-token completion from a pinned chat model."""

    def __init__(
        self, name: str, provider: ProviderName, model: str, required: bool, settings: AppConfig
    ) -> None:
        super().__init__(name, settings.llm.provider_settings(provider).base_url, model, required)
        self._provider = provider
        self._check = settings.llm.check

    async def probe(self) -> ProbeResult:
        return await self.timed(self._call)

    async def _call(self) -> ProbeResult:
        started = time.perf_counter()
        client = create_pinned_llm_client(self._provider, self._model)
        result = await client.complete(
            CompletionRequest(
                messages=[Message(role="user", content=self._check.prompt)],
                max_tokens=self._check.max_tokens,
            )
        )
        return self.result("ok", started, detail=result.content.strip())


class EmbeddingProbe(_TimedProbe):
    """One embedding; a silent fallback to the local embedder counts as failure."""

    def __init__(self, settings: AppConfig) -> None:
        super().__init__(
            "embeddings",
            settings.llm.provider_settings(settings.llm.embedding.provider).base_url,
            settings.llm.embedding.model,
            required=True,
        )
        self._text = settings.llm.check.prompt

    async def probe(self) -> ProbeResult:
        return await self.timed(self._call)

    async def _call(self) -> ProbeResult:
        started = time.perf_counter()
        client = HostedEmbeddingClient()
        vectors = await client.embed([self._text])
        if client.model_id != self._model:
            return self.result("failed", started, detail="fell back to the local embedder")
        return self.result("ok", started, detail=f"{len(vectors[0])} dimensions")


GuardFactory = Callable[
    [SecurityGuardModelSettings], HostedContentSafetyGuard | HostedJailbreakGuard
]


class GuardProbe(_TimedProbe):
    """One classification from a hosted security guard."""

    def __init__(
        self,
        name: str,
        settings: SecurityGuardModelSettings,
        required: bool,
        factory: GuardFactory,
        text: str,
    ) -> None:
        super().__init__(name, f"{settings.base_url}{settings.endpoint}", settings.model, required)
        self._guard = factory(settings)
        self._text = text

    async def probe(self) -> ProbeResult:
        return await self.timed(self._call)

    async def _call(self) -> ProbeResult:
        started = time.perf_counter()
        verdict: GuardVerdict = await self._guard.classify(self._text)
        if not verdict.available:
            return self.result(
                "failed", started, http_status=verdict.http_status, detail=verdict.reason
            )
        return self.result("ok", started, detail=verdict.decision)


def guard_probe(config: AppConfig, model: str, required: bool) -> EndpointProbe:
    """The probe for the security guard that names `model` (jailbreak or content safety)."""
    guards = config.security.guards
    for name, factory in (
        ("jailbreak", _jailbreak_guard),
        ("content_safety", _content_safety_guard),
    ):
        settings: SecurityGuardModelSettings = getattr(guards, name)
        if settings.model == model:
            return GuardProbe(model, settings, required, factory, config.llm.check.prompt)
    raise KeyError(f"no security guard uses '{model}'")


def _content_safety_guard(settings: SecurityGuardModelSettings) -> HostedContentSafetyGuard:
    return HostedContentSafetyGuard(
        base_url=settings.base_url,
        endpoint=settings.endpoint,
        timeout_seconds=settings.timeout_seconds,
        model=settings.model,
        max_tokens=settings.max_tokens,
        resilient=ResilientCaller("probe-content-safety"),
    )


def _jailbreak_guard(settings: SecurityGuardModelSettings) -> HostedJailbreakGuard:
    return HostedJailbreakGuard(
        base_url=settings.base_url,
        endpoint=settings.endpoint,
        timeout_seconds=settings.timeout_seconds,
        model=settings.model,
        resilient=ResilientCaller("probe-jailbreak"),
    )


def _status_of(exc: Exception) -> int | None:
    status = getattr(exc, "status_code", None)
    return status if isinstance(status, int) else None
