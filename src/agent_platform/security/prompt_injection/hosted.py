"""Hosted jailbreak/prompt-injection model adapter: who it calls is configuration."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Protocol, runtime_checkable

import httpx

from agent_platform.config import get_settings
from agent_platform.config.settings import SecurityGuardModelSettings
from agent_platform.logging import get_logger
from agent_platform.resilience import ResilientCaller, dependency_failure_fields
from agent_platform.security.guard_types import GuardVerdict
from agent_platform.security.remote_http import post_json_with_poll


@runtime_checkable
class JailbreakGuard(Protocol):
    """Detect prompt-injection/jailbreak attempts."""

    async def classify(self, text: str) -> GuardVerdict: ...


class HostedJailbreakGuard:
    """A hosted jailbreak classifier (`security.guards.jailbreak`), with bounded retries."""

    def __init__(
        self,
        *,
        base_url: str,
        endpoint: str,
        timeout_seconds: float,
        model: str,
        client: httpx.AsyncClient | None = None,
        resilient: ResilientCaller | None = None,
        settings: SecurityGuardModelSettings | None = None,
    ) -> None:
        self._settings = settings or get_settings().security.guards.jailbreak
        self._base_url = base_url.rstrip("/")
        self._endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        self._timeout = timeout_seconds
        self._model = model
        self._client = client
        self._resilient = resilient or ResilientCaller(self._settings.resilience_key)
        self._logger = get_logger("prompt_injection")

    @property
    def model(self) -> str:
        return self._model

    async def classify(self, text: str) -> GuardVerdict:
        """Classify text; 503/429/5xx failures are retried before unavailable."""
        started = time.perf_counter()
        api_key = self._settings.api_key
        if not api_key:
            return GuardVerdict(
                decision="unavailable",
                model=self._model,
                provider=self._settings.provider,
                reason=self._settings.missing_key_reason.format(
                    env=" or ".join(self._settings.key_env_vars)
                ),
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
                idempotency_key=_call_key(self._settings.resilience_key, text),
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
                provider=self._settings.provider,
                reason=(
                    self._settings.flagged_reason if flagged else self._settings.allowed_reason
                ),
                endpoint=self._endpoint,
                api_called=True,
                latency_ms=latency_ms,
                raw=data,
            )
        except Exception as exc:
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            error_fields = dependency_failure_fields(exc)
            self._logger.warning(
                "Hosted jailbreak guard unavailable after retries",
                model=self._model,
                latency_ms=latency_ms,
                attempts=self._resilient.attempts,
                timeout_seconds=self._timeout,
                **error_fields,
            )
            return GuardVerdict(
                decision="unavailable",
                model=self._model,
                provider=self._settings.provider,
                reason=str(error_fields["reason"]),
                error_code=str(error_fields["error_code"]),
                http_status=error_fields["http_status"],
                endpoint=self._endpoint,
                api_called=True,
                latency_ms=latency_ms,
            )


class ChatJailbreakGuard:
    """A chat model prompted as a jailbreak classifier — a guard's fallback.

    Asked the configured `classifier_prompt` over the latest message; its JSON answer is read
    at `verdict_key` and `score_key`. An answer that is not that JSON is *unavailable*, not a
    pass: a fallback that cannot classify must not wave a message through.
    """

    def __init__(
        self,
        settings: SecurityGuardModelSettings,
        client: httpx.AsyncClient | None = None,
        resilient: ResilientCaller | None = None,
    ) -> None:
        self._settings = settings
        self._client = client
        self._resilient = resilient or ResilientCaller(settings.resilience_key)
        self._logger = get_logger("prompt_injection")

    @property
    def model(self) -> str:
        return self._settings.model

    async def classify(self, text: str) -> GuardVerdict:
        settings = self._settings
        started = time.perf_counter()
        payload = {
            "model": settings.model,
            "messages": [
                {"role": "system", "content": settings.classifier_prompt},
                {"role": "user", "content": text},
            ],
            "max_tokens": settings.max_tokens,
            "temperature": 0,
            "stream": False,
            **settings.extra_body,
        }
        try:
            data = await post_json_with_poll(
                base_url=settings.base_url.rstrip("/"),
                endpoint=settings.endpoint,
                payload=payload,
                api_key=settings.api_key,
                timeout=settings.timeout_seconds,
                client=self._client,
                resilient=self._resilient,
                idempotency_key=_call_key(settings.resilience_key, text),
                error_label="jailbreak-fallback",
            )
            answer = _json_answer(data)
            flagged = bool(answer[settings.verdict_key])
        except Exception as exc:
            return self._unavailable(exc, started)
        return GuardVerdict(
            decision="flagged" if flagged else "allow",
            flagged=flagged,
            score=_as_float(answer.get(settings.score_key)),
            model=settings.model,
            provider=settings.provider,
            reason=settings.flagged_reason if flagged else settings.allowed_reason,
            endpoint=settings.endpoint,
            api_called=True,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            raw=data,
        )

    def _unavailable(self, exc: Exception, started: float) -> GuardVerdict:
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        error_fields = dependency_failure_fields(exc)
        self._logger.warning(
            "Jailbreak fallback unavailable",
            model=self._settings.model,
            latency_ms=latency_ms,
            **error_fields,
        )
        return GuardVerdict(
            decision="unavailable",
            model=self._settings.model,
            provider=self._settings.provider,
            reason=str(error_fields["reason"]),
            error_code=str(error_fields["error_code"]),
            http_status=error_fields["http_status"],
            endpoint=self._settings.endpoint,
            api_called=True,
            latency_ms=latency_ms,
        )


def _json_answer(data: dict[str, Any]) -> dict[str, Any]:
    """The JSON object a chat classifier answered with (a fenced block is unwrapped)."""
    content = str(data["choices"][0]["message"].get("content") or "").strip()
    start, end = content.find("{"), content.rfind("}")
    parsed = json.loads(content[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("the classifier did not answer with a JSON object")
    return parsed


def _as_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _call_key(prefix: str, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}:{digest}"
