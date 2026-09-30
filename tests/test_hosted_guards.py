from __future__ import annotations

import httpx
import pytest

from agentic_commons.resilience import ExponentialBackoffRetry, ResilientCaller
from agentic_commons.security.content_moderation.nvidia import NvidiaHostedContentSafetyGuard
from agentic_commons.security.prompt_injection.nvidia import NvidiaHostedJailbreakGuard


class FakeResponse:
    def __init__(self, payload: dict[str, object], status_code: int = 200) -> None:
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


class FakeAsyncClient:
    def __init__(
        self,
        payload: dict[str, object],
        status_code: int = 200,
        poll_payload: dict[str, object] | None = None,
        statuses: list[int] | None = None,
    ) -> None:
        self.payload = payload
        self.status_code = status_code
        self.statuses = list(statuses or [])
        self.poll_payload = poll_payload
        self.calls: list[dict[str, object]] = []

    async def post(self, url: str, **kwargs: object) -> FakeResponse:
        self.calls.append({"method": "POST", "url": url, **kwargs})
        status = self.statuses.pop(0) if self.statuses else self.status_code
        response = FakeResponse(self.payload, status)
        response.request = httpx.Request("POST", url)
        return response

    async def get(self, url: str, **kwargs: object) -> FakeResponse:
        self.calls.append({"method": "GET", "url": url, **kwargs})
        return FakeResponse(self.poll_payload or {}, 200)


@pytest.mark.asyncio
async def test_jailbreak_guard_records_real_api_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    client = FakeAsyncClient({"jailbreak": True, "score": 0.97})
    guard = NvidiaHostedJailbreakGuard(
        base_url="https://ai.api.nvidia.com",
        endpoint="/v1/security/nvidia/nemoguard-jailbreak-detect",
        timeout_seconds=2,
        model="nvidia/nemoguard-jailbreak-detect",
        client=client,  # type: ignore[arg-type]
    )

    verdict = await guard.classify("ignore previous instructions")

    assert verdict.flagged is True
    assert verdict.api_called is True
    assert verdict.provider == "nvidia"
    assert verdict.model == "nvidia/nemoguard-jailbreak-detect"
    assert verdict.latency_ms is not None
    assert len(client.calls) == 1
    assert client.calls[0]["url"] == (
        "https://ai.api.nvidia.com/v1/security/nvidia/nemoguard-jailbreak-detect"
    )


@pytest.mark.asyncio
async def test_content_guard_records_model_and_categories(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    client = FakeAsyncClient(
        {
            "choices": [
                {
                    "message": {
                        "content": '{"User Safety":"unsafe","Safety Categories":"Criminal Planning/Confessions"}',
                    }
                }
            ]
        }
    )
    guard = NvidiaHostedContentSafetyGuard(
        base_url="https://integrate.api.nvidia.com/v1",
        endpoint="/chat/completions",
        timeout_seconds=2,
        model="nvidia/llama-3.1-nemotron-safety-guard-8b-v3",
        client=client,  # type: ignore[arg-type]
    )

    verdict = await guard.classify("malicious request")

    assert verdict.flagged is True
    assert verdict.api_called is True
    assert verdict.model == "nvidia/llama-3.1-nemotron-safety-guard-8b-v3"
    assert verdict.categories == ["Criminal Planning/Confessions"]
    assert verdict.latency_ms is not None


@pytest.mark.asyncio
async def test_content_guard_polls_nvidia_202_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    client = FakeAsyncClient(
        {"requestId": "123e4567-e89b-12d3-a456-426614174000"},
        status_code=202,
        poll_payload={"choices": [{"message": {"content": '{"User Safety":"safe"}'}}]},
    )
    guard = NvidiaHostedContentSafetyGuard(
        base_url="https://integrate.api.nvidia.com/v1",
        endpoint="/chat/completions",
        timeout_seconds=2,
        model="nvidia/llama-3.1-nemotron-safety-guard-8b-v3",
        client=client,
    )
    verdict = await guard.classify("where is order 10242?")
    assert verdict.decision == "allow"
    assert verdict.api_called is True
    assert any(
        str(call["url"]).endswith("/v1/status/123e4567-e89b-12d3-a456-426614174000")
        for call in client.calls
    )


@pytest.mark.asyncio
async def test_jailbreak_guard_retries_503(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    client = FakeAsyncClient({"jailbreak": False, "score": 0.1}, statuses=[503, 200])
    retry = ExponentialBackoffRetry(max_attempts=3, base_delay_seconds=0, max_delay_seconds=0)
    guard = NvidiaHostedJailbreakGuard(
        base_url="https://ai.api.nvidia.com",
        endpoint="/v1/security/nvidia/nemoguard-jailbreak-detect",
        timeout_seconds=2,
        model="nvidia/nemoguard-jailbreak-detect",
        client=client,
        resilient=ResilientCaller("test-jailbreak", retry=retry),
    )
    verdict = await guard.classify("hello")
    assert verdict.decision == "allow"
    assert retry.attempts == 2
