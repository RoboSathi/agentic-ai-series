"""Endpoint probes for `make llm-check` — one per hosted model the repo calls.

Each probe sends the smallest *real* request its adapter can make,
through the same adapter the agent uses (and so through the rate
limiter). Listing models is not proof: NIM entitlements are per account,
so a model can be listed and still refuse to answer.

Probes are built from configuration, never from a hardcoded list: add a
hosted model to config and its probe appears; switch a section off and
its probe reports `skipped`.
"""

import asyncio
import time
from typing import Awaitable, Callable, Literal, Protocol, runtime_checkable

from pydantic import BaseModel

from agentic_commons.config import AppConfig, get_settings
from agentic_commons.config.settings import ProviderName, SecurityGuardModelSettings
from agentic_commons.llm.client import create_pinned_llm_client
from agentic_commons.llm.embeddings import NimEmbeddingClient
from agentic_commons.llm.types import CompletionRequest, Message
from agentic_commons.resilience import ResilientCaller
from agentic_commons.security.content_moderation.nvidia import NvidiaHostedContentSafetyGuard
from agentic_commons.security.guard_types import GuardVerdict
from agentic_commons.security.prompt_injection.nvidia import NvidiaHostedJailbreakGuard

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
            async with asyncio.timeout(self._timeout):
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
            settings.llm.provider_settings("nim").base_url,
            settings.llm.embedding.nim_model,
            required=True,
        )
        self._text = settings.llm.check.prompt

    async def probe(self) -> ProbeResult:
        return await self.timed(self._call)

    async def _call(self) -> ProbeResult:
        started = time.perf_counter()
        client = NimEmbeddingClient()
        vectors = await client.embed([self._text])
        if client.model_id != self._model:
            return self.result("failed", started, detail="fell back to the local embedder")
        return self.result("ok", started, detail=f"{len(vectors[0])} dimensions")


GuardFactory = Callable[
    [SecurityGuardModelSettings], NvidiaHostedContentSafetyGuard | NvidiaHostedJailbreakGuard
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


class SkippedProbe:
    """An endpoint configuration switched off — reported, not called."""

    def __init__(self, name: str, target: str, model: str, reason: str) -> None:
        self.name = name
        self._result = ProbeResult(
            name=name, target=target, model=model, status="skipped", required=False, detail=reason
        )

    async def probe(self) -> ProbeResult:
        return self._result


def build_probes(settings: AppConfig | None = None) -> list[EndpointProbe]:
    """Every hosted endpoint the configuration names, in a stable order."""
    config = settings or get_settings()
    return [
        _agent_chat_probe(config),
        _judge_probe(config),
        *_role_probes(config),
        _embedding_probe(config),
        _guard_probe(config, "content_safety", "content_moderation", _content_safety_guard),
        _guard_probe(config, "jailbreak", "prompt_injection", _jailbreak_guard),
    ]


def _role_probes(config: AppConfig) -> list[EndpointProbe]:
    """One probe per distinct model the running agent's roles name, labelled with its roles.

    The roles are the agent's (its overlay declares them); a model already probed as the
    platform's chat model or the eval judge is probed again here under its role names, so a
    role whose model this account cannot invoke is visible as that role.
    """
    by_model: dict[tuple[ProviderName, str], list[str]] = {}
    for role, spec in sorted(config.llm.roles.items()):
        by_model.setdefault((spec.provider, spec.model), []).append(role)
    return [
        ChatProbe(f"roles: {', '.join(roles)}", provider, model, config.llm.enabled, config)
        for (provider, model), roles in by_model.items()
    ]


def _agent_chat_probe(config: AppConfig) -> EndpointProbe:
    provider = config.llm.provider
    model = config.llm.provider_settings(provider).model
    return ChatProbe("agent_chat", provider, model, config.llm.enabled, config)


def _judge_probe(config: AppConfig) -> EndpointProbe:
    judge = config.evals.judge
    return ChatProbe("eval_judge", judge.provider, judge.model, config.evals.enabled, config)


def _embedding_probe(config: AppConfig) -> EndpointProbe:
    embedding = config.llm.embedding
    if embedding.provider != "nim":
        return SkippedProbe(
            "embeddings", "local", embedding.nim_model, "embedding.provider is local"
        )
    return EmbeddingProbe(config)


def _guard_probe(
    config: AppConfig, name: str, feature: str, factory: GuardFactory
) -> EndpointProbe:
    guards = config.security.guards
    guard: SecurityGuardModelSettings = getattr(guards, name)
    if not (guards.enabled and guard.enabled):
        return SkippedProbe(name, guard.base_url, guard.model, "guard disabled in config")
    # Reachability is checked either way; it is only *required* when the
    # agent's declared posture actually uses it.
    required = config.features.is_on(feature)
    return GuardProbe(name, guard, required, factory, config.llm.check.prompt)


def _content_safety_guard(settings: SecurityGuardModelSettings) -> NvidiaHostedContentSafetyGuard:
    return NvidiaHostedContentSafetyGuard(
        base_url=settings.base_url,
        endpoint=settings.endpoint,
        timeout_seconds=settings.timeout_seconds,
        model=settings.model,
        max_tokens=settings.max_tokens,
        resilient=ResilientCaller("probe-content-safety"),
    )


def _jailbreak_guard(settings: SecurityGuardModelSettings) -> NvidiaHostedJailbreakGuard:
    return NvidiaHostedJailbreakGuard(
        base_url=settings.base_url,
        endpoint=settings.endpoint,
        timeout_seconds=settings.timeout_seconds,
        model=settings.model,
        resilient=ResilientCaller("probe-jailbreak"),
    )


def _status_of(exc: Exception) -> int | None:
    status = getattr(exc, "status_code", None)
    return status if isinstance(status, int) else None
