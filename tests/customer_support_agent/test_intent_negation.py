"""A phrase counts only when it is asked for: "I don't want a refund" is not a refund request."""

import pytest

from agent_platform.intent import IntentCandidate
from customer_support_agent.intent.phrases import PhraseReader
from customer_support_agent.intent.router import CustomerSupportIntentRouter


class _Backend:
    """A classifier that is sure every message is a refund request."""

    model_name = "scripted"
    score_type = "model_softmax"
    is_loaded = True

    def __init__(self) -> None:
        self.calls = 0

    async def ensure_loaded(self) -> None:
        return None

    async def classify(self, text: str) -> list[IntentCandidate]:
        self.calls += 1
        return [IntentCandidate(intent="REQUEST_REFUND", score=0.99)]


class TestThePhraseReader:
    @pytest.mark.parametrize(
        "text",
        [
            "I don't want a refund",
            "i dont want refund",
            "I didn’t get my refund",
            "no refund please",
        ],
    )
    def test_a_negated_phrase_is_refused(self, text: str) -> None:
        reader = PhraseReader()
        assert reader.refused(text, "refund") and not reader.asserted(text, "refund")

    def test_an_asked_phrase_is_asserted(self) -> None:
        assert PhraseReader().asserted("Please refund order 10487", "refund")

    def test_asked_once_is_asserted_even_if_negated_elsewhere(self) -> None:
        text = "Don't cancel it, I just want a refund"
        assert PhraseReader().asserted(text, "refund")
        assert PhraseReader().refused(text, "cancel it")


class TestTheRouter:
    async def test_a_refused_refund_is_not_granted_by_the_classifier(self) -> None:
        router = CustomerSupportIntentRouter(backend=_Backend())
        intent = await router.classify_async("I don't want a refund for order 10487")
        assert intent.name == "UNKNOWN"
        assert "do not want REQUEST_REFUND" in intent.reason

    @pytest.mark.parametrize(
        "text",
        ["I didn't get my refund for order 10487", "i didnt get my refund", "still no refund"],
    )
    async def test_a_missing_refund_asks_where_it_is(self, text: str) -> None:
        backend = _Backend()
        intent = await CustomerSupportIntentRouter(backend=backend).classify_async(text)
        assert intent.name == "REFUND_STATUS" and backend.calls == 0

    async def test_a_refund_asked_for_is_still_a_refund_request(self) -> None:
        intent = await CustomerSupportIntentRouter(backend=_Backend()).classify_async(
            "I want a refund for order 10487"
        )
        assert intent.name == "REQUEST_REFUND"
