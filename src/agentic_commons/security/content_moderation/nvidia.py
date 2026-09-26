"""NVIDIA-hosted content-safety model adapter."""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Protocol, runtime_checkable

import httpx

from agentic_commons.config import get_settings
from agentic_commons.logging import get_logger
from agentic_commons.resilience import ResilientCaller, dependency_failure_fields
from agentic_commons.security.guard_types import GuardVerdict
from agentic_commons.security.remote_http import post_json_with_poll


@runtime_checkable
class ContentSafetyGuard(Protocol):
    """Classify unsafe content in user or assistant text."""

    async def classify(self, text: str) -> GuardVerdict: ...


class NvidiaHostedContentSafetyGuard:
    """NVIDIA-hosted Nemotron Safety Guard v3 client with bounded retries."""

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
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        self._timeout = timeout_seconds
        self._model = model
        self._max_tokens = max(16, min(max_tokens, 128))
        self._client = client
        self._resilient = resilient or ResilientCaller("nvidia-content-safety")
        self._logger = get_logger("content_moderation")

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
                idempotency_key=_call_key(text),
                error_label="content-safety",
            )
            output = _extract_chat_content(data)
            flagged, categories = _parse_nemotron_safety_output(output)
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            return GuardVerdict(
                decision="flagged" if flagged else "allow",
                flagged=flagged,
                categories=categories,
                model=self._model,
                provider="nvidia",
                reason=(
                    "NVIDIA Nemotron Safety Guard flagged the input."
                    if flagged
                    else "NVIDIA Nemotron Safety Guard classified the input as safe."
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
                provider="nvidia",
                reason=str(error_fields["reason"]),
                error_code=str(error_fields["error_code"]),
                http_status=error_fields["http_status"],
                endpoint=self._endpoint,
                api_called=True,
                latency_ms=latency_ms,
            )


def _nvidia_api_key() -> str:
    return os.environ.get("nvidia_api_key") or os.environ.get("NVIDIA_API_KEY", "")


def _extract_chat_content(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("NVIDIA content-safety response contained no choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise ValueError("NVIDIA content-safety response contained no text classification")
    return content.strip()


def _safety_system_prompt() -> str:
    """What the classifier is protecting, per configuration.

    A literal here said this platform classifies for a customer-support
    application, which is a claim about the agent, not about content
    safety. An agent that protects something else overrides it.
    """
    return get_settings().security.content_moderation.system_prompt


def _parse_nemotron_safety_output(output: str) -> tuple[bool, list[str]]:
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
            safety = str(parsed.get("User Safety", "")).strip().casefold()
            categories_raw = parsed.get("Safety Categories", "")
            categories = [c.strip() for c in str(categories_raw).split(",") if c.strip()]
            if safety in {"safe", "unsafe"}:
                return safety == "unsafe", categories
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("NVIDIA content-safety guard returned an empty classification")
    first = lines[0].casefold()
    flagged = "unsafe" in first
    categories = [token.strip(" ,:;[]()") for token in lines[1:] if token.strip(" ,:;[]()")]
    return flagged, categories


def _call_key(text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]
    return f"nvidia-content-safety:{digest}"
