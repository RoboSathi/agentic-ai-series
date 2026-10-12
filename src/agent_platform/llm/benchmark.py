"""`make model-check BENCH=1`: how fast each chat model the configuration names really is.

One minimal probe says a model answers; it says little about how long a real call takes. The
benchmark sends the same judge-sized request `samples` times to each distinct chat model (the
platform's, the eval judge, every agent role) and reports the median and 95th-percentile
latency and the completion throughput — the numbers to choose a judge or an answerer by.

Calls to one model run one after another, so the numbers are per-call, not a contended burst;
different models run side by side. Every call goes through the same adapter (and so the same
rate limiter) the agent uses.
"""

import asyncio
import json
import statistics
import time
from typing import Any

from pydantic import BaseModel, Field

from agent_platform.config import AppConfig, get_settings
from agent_platform.config.settings import ProviderName
from agent_platform.llm.client import create_pinned_llm_client
from agent_platform.llm.types import CompletionRequest, Message
from agent_platform.logging import describe_error

_PERCENTILE_95 = 0.95


class ChatModel(BaseModel):
    """One distinct chat model (with its request fields), and every name it is used under."""

    provider: ProviderName
    model: str
    # A role's provider-specific request fields (thinking off): measured as its own row.
    extra_body: dict[str, Any] = Field(default_factory=dict)
    uses: list[str] = Field(default_factory=list)


class BenchmarkResult(BaseModel):
    model: str
    extra_body: dict[str, Any] = Field(default_factory=dict)
    uses: list[str]
    samples: int
    failures: int = 0
    p50_seconds: float = 0.0
    p95_seconds: float = 0.0
    completion_tokens_per_second: float = 0.0
    detail: str = ""


def chat_models(settings: AppConfig | None = None) -> list[ChatModel]:
    """The platform model, the eval judge and every agent role, one entry per model."""
    config = settings or get_settings()
    found: dict[tuple[str, str, str], ChatModel] = {}

    def add(provider: ProviderName, model: str, use: str, extra: dict[str, Any]) -> None:
        key = (provider, model, json.dumps(extra, sort_keys=True))
        entry = found.setdefault(key, ChatModel(provider=provider, model=model, extra_body=extra))
        entry.uses.append(use)

    add(config.llm.provider, config.llm.provider_settings().model, "agent_chat", {})
    add(config.judge.provider, config.judge.model, "judge", config.judge.extra_body)
    for role, spec in sorted(config.llm.roles.items()):
        add(spec.provider, spec.model, f"role:{role}", spec.extra_body)
    return list(found.values())


async def benchmark(
    settings: AppConfig | None = None, models: list[ChatModel] | None = None
) -> list[BenchmarkResult]:
    """Every chat model (`models`, or the configuration's), measured side by side."""
    config = settings or get_settings()
    targets = models if models is not None else chat_models(config)
    return list(await asyncio.gather(*(_measure(model, config) for model in targets)))


async def _measure(target: ChatModel, config: AppConfig) -> BenchmarkResult:
    bench = config.llm.check.benchmark
    client = create_pinned_llm_client(target.provider, target.model)
    request = CompletionRequest(
        messages=[Message(role="user", content=bench.prompt)],
        max_tokens=bench.max_tokens,
        extra_body=target.extra_body,
    )
    latencies: list[float] = []
    tokens = 0
    errors: list[str] = []
    for _ in range(bench.samples):
        started = time.perf_counter()
        try:
            async with asyncio.timeout(bench.timeout_seconds):
                result = await client.complete(request)
        except Exception as exc:
            errors.append(describe_error(exc) if not isinstance(exc, TimeoutError) else "timeout")
            continue
        latencies.append(time.perf_counter() - started)
        tokens += result.usage.completion_tokens
    return _summary(target, bench.samples, latencies, tokens, errors)


def _summary(
    target: ChatModel, samples: int, latencies: list[float], tokens: int, errors: list[str]
) -> BenchmarkResult:
    base = BenchmarkResult(
        model=target.model,
        extra_body=target.extra_body,
        uses=target.uses,
        samples=samples,
        failures=len(errors),
        detail="; ".join(dict.fromkeys(errors)),
    )
    if not latencies:
        return base
    ordered = sorted(latencies)
    p95 = ordered[min(len(ordered) - 1, int(round(_PERCENTILE_95 * (len(ordered) - 1))))]
    return base.model_copy(
        update={
            "p50_seconds": round(statistics.median(ordered), 2),
            "p95_seconds": round(p95, 2),
            "completion_tokens_per_second": round(tokens / sum(latencies), 1),
        }
    )


__all__ = ["BenchmarkResult", "ChatModel", "benchmark", "chat_models"]
