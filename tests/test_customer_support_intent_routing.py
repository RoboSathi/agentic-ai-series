from __future__ import annotations

import pytest

from agent_platform.intent import IntentCandidate
from customer_support_agent.intent.router import CustomerSupportIntentRouter


class FakeBackend:
    model_name = "test-intent-classifier"
    score_type = "model_softmax"
    is_loaded = True

    def __init__(self, candidates: list[IntentCandidate]) -> None:
        self._candidates = candidates
        self.calls = 0

    async def ensure_loaded(self) -> None:
        return None

    async def classify(self, text: str) -> list[IntentCandidate]:
        self.calls += 1
        return list(self._candidates)


def _backend(*items: tuple[str, float]) -> FakeBackend:
    return FakeBackend([IntentCandidate(intent=name, score=score) for name, score in items])


@pytest.mark.asyncio
async def test_obvious_order_status_is_deterministic_and_skips_model() -> None:
    backend = _backend(("fallback", 0.90), ("track_order", 0.10))
    router = CustomerSupportIntentRouter(backend=backend)

    result = await router.classify_async("where is the order 10242 now?")

    assert result.name == "ORDER_STATUS"
    assert result.source == "deterministic"
    assert result.confidence == 1.0
    assert result.allowed_tools == ["get_order_status"]
    assert backend.calls == 0


@pytest.mark.asyncio
async def test_refund_status_beats_generic_refund_marker() -> None:
    backend = _backend(("get_refund", 0.90))
    router = CustomerSupportIntentRouter(backend=backend)

    result = await router.classify_async("where is my refund?")

    assert result.name == "REFUND_STATUS"
    assert result.source == "deterministic"
    assert result.allowed_tools == ["get_refund_status"]
    assert backend.calls == 0


def test_intent_operator_fields_are_concise_and_distinguish_source() -> None:
    from customer_support_agent.graph.agent import CustomerSupportAgent

    class DeterministicIntent:
        name = "ORDER_STATUS"
        source = "deterministic"
        matched_phrase = "phrase:order status"
        model = ""
        confidence = 1.0
        disposition = type("Disposition", (), {"value": "supported"})()
        reason = ""
        clarification = ""
        candidates = []

    deterministic_fields = CustomerSupportAgent._intent_status_fields(DeterministicIntent())
    assert deterministic_fields == {
        "intent": "ORDER_STATUS",
        "classification": "deterministic",
        "matched": "order status",
    }

    class BertIntent:
        name = "ORDER_STATUS"
        source = "huggingface_local"
        matched_phrase = ""
        model = "bert-test-model"
        confidence = 0.8437
        disposition = type("Disposition", (), {"value": "supported"})()
        reason = ""
        clarification = ""
        candidates = []

    bert_fields = CustomerSupportAgent._intent_status_fields(BertIntent())
    assert bert_fields == {
        "intent": "ORDER_STATUS",
        "classification": "bert",
        "model": "bert-test-model",
        "score": 0.844,
    }
