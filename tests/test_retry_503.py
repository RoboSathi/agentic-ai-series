from __future__ import annotations

import httpx
import pytest

from agentic_commons.resilience import ExponentialBackoffRetry


@pytest.mark.asyncio
async def test_503_is_retried_until_success() -> None:
    retry = ExponentialBackoffRetry(max_attempts=3, base_delay_seconds=0, max_delay_seconds=0)
    calls = 0

    async def operation() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            request = httpx.Request("POST", "https://example.invalid")
            response = httpx.Response(503, request=request)
            raise httpx.HTTPStatusError(
                "503 service unavailable", request=request, response=response
            )
        return "ok"

    assert await retry.execute(operation, "test:503") == "ok"
    assert calls == 3
