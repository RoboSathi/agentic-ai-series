"""Hosted content-safety model adapter: who it calls, and how it is worded, is configuration."""

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
class ContentSafetyGuard(Protocol):
    """Classify unsafe content in user or assistant text."""

    async def classify(self, text: str) -> GuardVerdict: ...


class HostedContentSafetyGuard:
    """A hosted content-safety classifier (`security.guards.content_safety`), bounded retries."""

    def __init__(
        self,
        *,
        base_url: str,
        endpoint: str,
        timeout_seconds: float,
        model: str,
        max_tokens: int = 128,
        client: httpx.AsyncClient | None = None,
        resilient: ResilientCaller | None = None,
        settings: SecurityGuardModelSettings | None = None,
    ) -> None:
        self._settings = settings or get_settings().security.guards.content_safety
        self._base_url = base_url.rstrip("/")
        self._endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        self._timeout = timeout_seconds
        self._model = model
        self._max_tokens = max(16, min(max_tokens, 128))
        self._client = client
        self._resilient = resilient or ResilientCaller(self._settings.resilience_key)
        self._logger = get_logger("content_moderation")

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

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _safety_system_prompt()},
                {"role": "user", "content": text},
            ],
            "max_tokens": self._max_tokens,
            "temperature": 0,
            "stream": False,
        }
        try:
            data = await post_json_with_poll(
                base_url=self._base_url,
                endpoint=self._endpoint,
                payload=payload,
                api_key=api_key,
                timeout=self._timeout,
                client=self._client,
                resilient=self._resilient,
                idempotency_key=_call_key(self._settings.resilience_key, text),
                error_label="content-safety",
            )
            output = _extract_chat_content(data)
            flagged, categories = _parse_safety_output(output)
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            return GuardVerdict(
                decision="flagged" if flagged else "allow",
                flagged=flagged,
                categories=categories,
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
                "Hosted content-safety guard unavailable after retries",
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


def _extract_chat_content(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("content-safety response contained no choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise ValueError("content-safety response contained no text classification")
    return content.strip()


def _safety_system_prompt() -> str:
    """What the classifier is protecting, per configuration.

    A literal here said this platform classifies for a customer-support
    application, which is a claim about the agent, not about content
    safety. An agent that protects something else overrides it.
    """
    return get_settings().security.content_moderation.system_prompt


def _parse_safety_output(output: str) -> tuple[bool, list[str]]:
    """(flagged, categories) from the classifier's answer, read with the configured keys."""
    words = get_settings().security.content_moderation
    text = output.strip()
    candidates = [text]
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            candidates.insert(0, parts[1].partition("\n")[2].strip())
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            safety = str(parsed.get(words.response_safety_key, "")).strip().casefold()
            categories_raw = parsed.get(words.response_categories_key, "")
            categories = [c.strip() for c in str(categories_raw).split(",") if c.strip()]
            if safety in {words.safe_word, words.unsafe_word}:
                return safety == words.unsafe_word, categories
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("content-safety guard returned an empty classification")
    first = lines[0].casefold()
    flagged = words.unsafe_word in first
    categories = [token.strip(" ,:;[]()") for token in lines[1:] if token.strip(" ,:;[]()")]
    return flagged, categories


def _call_key(prefix: str, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}:{digest}"
