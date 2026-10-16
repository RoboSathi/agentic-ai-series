from __future__ import annotations

import pytest

from agent_platform.intent import IntentCandidate
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
async def test_refund_status_is_deterministic_read() -> None:
    router = CustomerSupportIntentRouter(backend=FakeBackend())

    result = await router.classify_async("what is the refund status for 10417?")

    assert result.name == "REFUND_STATUS"
    assert result.source == "deterministic"
    assert result.allowed_tools == ["get_refund_status"]


class StatusBackend(FakeBackend):
    """A classifier that places every turn as REFUND_STATUS."""

    async def classify(self, text: str) -> list[IntentCandidate]:
        return [IntentCandidate(intent="REFUND_STATUS", score=0.9)]


@pytest.mark.asyncio
async def test_a_bare_mention_of_a_refund_is_not_a_request_by_phrase() -> None:
    """ADR C-10: "refund" alone no longer routes to REQUEST_REFUND before the classifier."""
    router = CustomerSupportIntentRouter(backend=StatusBackend())

    result = await router.classify_async("will the refund appear on my card statement?")

    assert result.name == "REFUND_STATUS"
    assert result.source != "deterministic"


@pytest.mark.asyncio
async def test_a_refund_policy_question_is_a_refund_status_read() -> None:
    router = CustomerSupportIntentRouter(backend=FakeBackend())

    result = await router.classify_async("how long will the refund take to show on my card?")

    assert result.name == "REFUND_STATUS"
    assert result.allowed_tools == ["get_refund_status"]


@pytest.mark.asyncio
async def test_asking_for_a_refund_by_order_is_still_a_request() -> None:
    router = CustomerSupportIntentRouter(backend=StatusBackend())

    result = await router.classify_async("refund order 10846")

    assert result.name == "REQUEST_REFUND"
    assert result.source == "deterministic"
