from __future__ import annotations

import pytest

from agentic_commons.intent import IntentCandidate
from customer_support_agent.intent.router import CustomerSupportIntentRouter


class FakeBackend:
    model_name = "test-intent-classifier"
    score_type = "model_softmax"
    is_loaded = True

    async def ensure_loaded(self) -> None:
        return None

    async def classify(self, text: str) -> list[IntentCandidate]:
        return [IntentCandidate(intent="refund", score=0.9)]


@pytest.mark.asyncio
async def test_bare_refund_is_ambiguous_not_write_authorized() -> None:
    router = CustomerSupportIntentRouter(backend=FakeBackend())

    result = await router.classify_async("refund")

    assert result.name == "UNKNOWN"
    assert result.disposition.value == "clarify"
    assert {candidate.intent for candidate in result.candidates} == {
        "REQUEST_REFUND",
        "REFUND_STATUS",
    }
    assert result.allowed_tools == []


@pytest.mark.asyncio
async def test_refund_status_is_deterministic_read() -> None:
    router = CustomerSupportIntentRouter(backend=FakeBackend())

    result = await router.classify_async("what is the refund status for 10417?")

    assert result.name == "REFUND_STATUS"
    assert result.source == "deterministic"
    assert result.allowed_tools == ["get_refund_status"]
