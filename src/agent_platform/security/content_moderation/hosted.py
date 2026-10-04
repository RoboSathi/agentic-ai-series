"""Hosted content-safety model adapter: who it calls, and how it is worded, is configuration."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Protocol, runtime_checkable

import httpx

from agent_platform.config import get_settings
from agent_platform.config.settings import ContentModerationSettings, SecurityGuardModelSettings
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

    def _reason(self, flagged: bool, categories: list[str], output: str) -> str:
        """What the verdict says: what the model flagged, in its own words — never a bare
        "flagged" that hides the evidence."""
        if not flagged:
            return self._settings.allowed_reason
        named = _flagged_as(self._settings.flagged_categories_reason, categories)
        if named:
            return named
        unnamed = self._settings.flagged_uncategorised_reason
        return unnamed.format(output=output) if unnamed else self._settings.flagged_reason

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
            **self._settings.extra_body,
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
                reason=self._reason(flagged, categories, output),
                endpoint=self._endpoint,
                api_called=True,
                latency_ms=latency_ms,
                output=output,
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


def _flagged_as(template: str, categories: list[str]) -> str:
    """The flagged reason naming what the model found, or "" when it named nothing."""
    return template.format(categories=", ".join(categories)) if template and categories else ""


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
            if safety in {words.safe_word, words.unsafe_word}:
                return safety == words.unsafe_word, _categories(
                    parsed, words.response_categories_keys
                )
    return _parse_plain_output(text, words)


def _parse_plain_output(text: str, words: ContentModerationSettings) -> tuple[bool, list[str]]:
    """(flagged, categories) from a `Key: value` answer, e.g. "User Safety: unsafe" then
    "Safety Categories: Violence, Sexual" — the model's native format."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("content-safety guard returned an empty classification")
    keyed = _keyed_lines(lines)
    safety = keyed.get(words.response_safety_key.casefold(), lines[0]).casefold()
    category_keys = [key.casefold() for key in words.response_categories_keys]
    return words.unsafe_word in safety, _categories(keyed, category_keys)


def _keyed_lines(lines: list[str]) -> dict[str, str]:
    """`Key: value` lines as {casefolded key: value}; lines without a colon are skipped."""
    pairs = (line.partition(":") for line in lines)
    return {key.strip().casefold(): value.strip() for key, colon, value in pairs if colon}


def _categories(parsed: dict[str, Any], keys: list[str]) -> list[str]:
    """The categories under the first key present, given as a string or as a list."""
    found = next((parsed[key] for key in keys if parsed.get(key)), "")
    items = found if isinstance(found, list) else str(found).split(",")
    return [str(item).strip() for item in items if str(item).strip()]


def _call_key(prefix: str, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}:{digest}"
