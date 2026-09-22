"""NVIDIA-hosted jailbreak/prompt-injection model adapter."""

from __future__ import annotations

import hashlib
import os
import time
from typing import Any, Protocol, runtime_checkable

import httpx

from agentic_commons.logging import get_logger
from agentic_commons.resilience import ResilientCaller
from agentic_commons.security.guard_types import GuardVerdict
from agentic_commons.security.remote_http import post_json_with_poll


@runtime_checkable
class JailbreakGuard(Protocol):
    """Detect prompt-injection/jailbreak attempts."""

    async def classify(self, text: str) -> GuardVerdict: ...


class NvidiaHostedJailbreakGuard:
    """NVIDIA-hosted NemoGuard JailbreakDetect client with bounded retries."""

    def __init__(
        self,
        *,
        base_url: str,
        endpoint: str,
        timeout_seconds: float,
        model: str,
        client: httpx.AsyncClient | None = None,
        resilient: ResilientCaller | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        self._timeout = timeout_seconds
        self._model = model
        self._client = client
        self._resilient = resilient or ResilientCaller("nvidia-jailbreak")
        self._logger = get_logger("security")

    @property
    def model(self) -> str:
        return self._model

    async def classify(self, text: str) -> GuardVerdict:
        """Classify text; 503/429/5xx failures are retried before unavailable."""
        started = time.perf_counter()
        api_key = _nvidia_api_key()
        if not api_key:
            return GuardVerdict(
                decision="unavailable",
                model=self._model,
                provider="nvidia",
                reason="nvidia_api_key is not configured",
                endpoint=self._endpoint,
                api_called=False,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )
        try:
            data = await post_json_with_poll(
                base_url=self._base_url,
                endpoint=self._endpoint,
                payload={"input": text},
                api_key=api_key,
                timeout=self._timeout,
                client=self._client,
                resilient=self._resilient,
                idempotency_key=_call_key(text),
                error_label="jailbreak",
            )
            flagged = bool(data.get("jailbreak", False))
            score = _as_float(data.get("score"))
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            return GuardVerdict(
                decision="flagged" if flagged else "allow",
                flagged=flagged,
                score=score,
                model=self._model,
                provider="nvidia",
                reason=(
                    "NVIDIA jailbreak classifier flagged the input."
                    if flagged
                    else "NVIDIA jailbreak classifier allowed the input."
                ),
                endpoint=self._endpoint,
                api_called=True,
                latency_ms=latency_ms,
                raw=data,
            )
        except Exception as exc:
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            self._logger.warning(
                "Hosted jailbreak guard unavailable after retries",
                model=self._model,
                error=str(exc),
                latency_ms=latency_ms,
                attempts=self._resilient.attempts,
            )
            return GuardVerdict(
                decision="unavailable",
                model=self._model,
                provider="nvidia",
                reason=str(exc),
                endpoint=self._endpoint,
                api_called=True,
                latency_ms=latency_ms,
            )


def _nvidia_api_key() -> str:
    return os.environ.get("nvidia_api_key") or os.environ.get("NVIDIA_API_KEY", "")


def _as_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _call_key(text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]
    return f"nvidia-jailbreak:{digest}"
