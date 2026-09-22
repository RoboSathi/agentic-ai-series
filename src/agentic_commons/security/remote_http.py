"""Resilient HTTP transport for hosted security-model APIs."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from agentic_commons.resilience import ResilientCaller


async def post_json_with_poll(
    *,
    base_url: str,
    endpoint: str,
    payload: dict[str, Any],
    api_key: str,
    timeout: float,
    client: httpx.AsyncClient | None,
    resilient: ResilientCaller,
    idempotency_key: str,
    error_label: str,
) -> dict[str, Any]:
    """POST JSON and resolve a provider-side asynchronous 202 result.

    The initial POST and every status GET are distinct external calls and are
    individually wrapped by the same resilience stack. HTTP 503/502/504 and
    429 responses are raised so the retry policy gets an opportunity to recover.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    owns_client = client is None
    active_client = client or httpx.AsyncClient(timeout=timeout)
    started = time.perf_counter()
    try:
        url = f"{base_url.rstrip('/')}{endpoint if endpoint.startswith('/') else '/' + endpoint}"

        async def post_once() -> httpx.Response:
            response = await active_client.post(
                url,
                json=payload,
                headers=headers,
                timeout=timeout,
            )
            if response.status_code != 202:
                response.raise_for_status()
            return response

        response = await resilient.call(
            post_once,
            idempotency_key=f"{idempotency_key}:post",
            timeout_seconds=timeout,
        )

        if response.status_code == 202:
            pending = response.json()
            request_id = pending.get("requestId") or pending.get("request_id")
            if not request_id:
                raise ValueError(f"NVIDIA {error_label} guard returned 202 without requestId")

            base = base_url.rstrip("/")
            poll_url = (
                f"{base}/status/{request_id}"
                if base.endswith("/v1")
                else f"{base}/v1/status/{request_id}"
            )
            deadline = started + timeout
            while time.perf_counter() < deadline:
                remaining = max(0.5, deadline - time.perf_counter())

                async def poll_once() -> httpx.Response:
                    poll = await active_client.get(
                        poll_url,
                        headers=headers,
                        timeout=remaining,
                    )
                    if poll.status_code != 202:
                        poll.raise_for_status()
                    return poll

                poll = await resilient.call(
                    poll_once,
                    idempotency_key=f"{idempotency_key}:poll:{request_id}",
                    timeout_seconds=remaining,
                )
                if poll.status_code == 200:
                    data = poll.json()
                    if not isinstance(data, dict):
                        raise ValueError(
                            f"NVIDIA {error_label} guard returned a non-object status response"
                        )
                    return data
                await asyncio.sleep(min(0.15, max(0.0, deadline - time.perf_counter())))

            raise TimeoutError(
                f"NVIDIA {error_label} guard result remained pending past {timeout:.1f}s"
            )

        data = response.json()
        if not isinstance(data, dict):
            raise ValueError(f"NVIDIA {error_label} guard returned a non-object response")
        return data
    finally:
        if owns_client:
            await active_client.aclose()
