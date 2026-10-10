from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_platform.intent import IntentCandidate, IntentDisposition
from customer_support_agent.config import get_agent_settings
from customer_support_agent.intent.router import CustomerSupportIntentRouter, Intent


class FakeBackend:
    model_name = "test-intent-classifier"
    score_type = "model_softmax"
    is_loaded = True

    def __init__(self, candidates: list[IntentCandidate]) -> None:
        self._candidates = candidates
        self.last_text = ""

    async def ensure_loaded(self) -> None:
        return None

    async def classify(self, text: str) -> list[IntentCandidate]:
        self.last_text = text
        return list(self._candidates)


def _settings() -> SimpleNamespace:
    intents = [
        SimpleNamespace(name="ORDER_STATUS", tools=["get_order_status"]),
        SimpleNamespace(name="ORDER_DETAILS", tools=["get_order_details"]),
        SimpleNamespace(name="REFUND_STATUS", tools=["get_refund_status"]),
        SimpleNamespace(name="REQUEST_REFUND", tools=["process_refund"]),
    ]
    labels = {item.name: [item.name] for item in intents}
    labels["UNKNOWN"] = ["UNKNOWN"]
    return SimpleNamespace(
        intents=intents,
        default_intent="UNKNOWN",
        followup=SimpleNamespace(
            phrases=["what about", "how about", "what of", "also", "same for", "and", "check"],
            reference_pattern=r"\b\d{4,}\b",
        ),
        intent_router=SimpleNamespace(
            enabled=True,
            clarification_top_k=2,
            phrase_rules=True,
            routing=lambda provider: SimpleNamespace(
                confidence_threshold=0.75, model_label_mapping=labels
            ),
            deterministic_rejects=[],
            negation=get_agent_settings().intent_router.negation,
        ),
    )


def _intent(name: str) -> Intent:
    return Intent(
        SimpleNamespace(
            intent=name,
            allowed_tools=[f"tool:{name}"],
            confidence=0.95,
            disposition=IntentDisposition.SUPPORTED,
            source="test",
            candidates=[IntentCandidate(intent=name, score=1.0)],
            matched_rule="",
            clarification="",
            reason="test",
            model="test",
            score_type="model_softmax",
        )
    )


def _router(
    monkeypatch: pytest.MonkeyPatch, candidates: list[IntentCandidate]
) -> CustomerSupportIntentRouter:
    monkeypatch.setattr("customer_support_agent.intent.router.get_agent_settings", _settings)
    return CustomerSupportIntentRouter(backend=FakeBackend(candidates))


def test_generic_single_word_followup_markers_are_not_context_triggers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = _router(
        monkeypatch,
        [IntentCandidate(intent="UNKNOWN", score=1.0)],
    )

    assert router._is_contextual_followup("and") is False
    assert router._is_contextual_followup("check") is False
    assert router._is_contextual_followup("what about it") is True
