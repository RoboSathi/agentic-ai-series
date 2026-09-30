from __future__ import annotations

import httpx
import pytest

from agentic_commons.resilience import ExponentialBackoffRetry, ResilientCaller
from agentic_commons.security.content_moderation.nvidia import NvidiaHostedContentSafetyGuard
from agentic_commons.security.prompt_injection.nvidia import NvidiaHostedJailbreakGuard


class FakeResponse:
    def __init__(self, payload: dict[str, object], status_code: int) -> None:
        self._payload = payload
        self.status_code = status_code
        self.request = httpx.Request("POST", "https://test.invalid")

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=self.request,
                response=httpx.Response(self.status_code, request=self.request),
            )

    def json(self) -> dict[str, object]:
        return self._payload


class FakeClient:
    def __init__(self, payload: dict[str, object]) -> None:
        self.statuses = [503, 200]
        self.payload = payload

    async def post(self, url: str, **kwargs: object) -> FakeResponse:
        status = self.statuses.pop(0)
        return FakeResponse(self.payload, status)


@pytest.mark.asyncio
async def test_jailbreak_guard_retries_503(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    client = FakeClient({"jailbreak": False, "score": 0.1})
    retry = ExponentialBackoffRetry(max_attempts=3, base_delay_seconds=0, max_delay_seconds=0)
    guard = NvidiaHostedJailbreakGuard(
        base_url="https://ai.api.nvidia.com",
        endpoint="/v1/security/nvidia/nemoguard-jailbreak-detect",
        timeout_seconds=2,
        model="nvidia/nemoguard-jailbreak-detect",
        client=client,  # type: ignore[arg-type]
        resilient=ResilientCaller("test-jailbreak", retry=retry),
    )
    verdict = await guard.classify("hello")
    assert verdict.decision == "allow"
    assert retry.attempts == 2


@pytest.mark.asyncio
async def test_content_guard_retries_503(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    client = FakeClient({"choices": [{"message": {"content": '{"User Safety":"safe"}'}}]})
    retry = ExponentialBackoffRetry(max_attempts=3, base_delay_seconds=0, max_delay_seconds=0)
    guard = NvidiaHostedContentSafetyGuard(
        base_url="https://integrate.api.nvidia.com/v1",
        endpoint="/chat/completions",
        timeout_seconds=2,
        model="nvidia/llama-3.1-nemotron-safety-guard-8b-v3",
        client=client,  # type: ignore[arg-type]
        resilient=ResilientCaller("test-content-safety", retry=retry),
    )
    verdict = await guard.classify("hello")
    assert verdict.decision == "allow"
    assert retry.attempts == 2
