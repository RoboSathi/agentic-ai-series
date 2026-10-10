"""CSA hands its own intents to whichever classifier the config names, and routes on the answer."""

from __future__ import annotations

from typing import Any

import pytest

from agent_platform.config import get_settings
from agent_platform.intent import IntentCandidate
from customer_support_agent.config import get_agent_settings
from customer_support_agent.intent import router as router_module
from customer_support_agent.intent.router import CustomerSupportIntentRouter


class IntentNamedBackend:
    """A classifier that already answers in CSA's own intent names, as Laya does."""

    model_name = "laya-test"
    score_type = "laya_calibrated"
    is_loaded = True

    def __init__(self, top: str, score: float) -> None:
        self._top = top
        self._score = score

    async def ensure_loaded(self) -> None:
        return None

    async def classify(self, text: str) -> list[IntentCandidate]:
        return [
            IntentCandidate(intent=self._top, score=self._score),
            IntentCandidate(intent="UNKNOWN", score=1.0 - self._score),
        ]


@pytest.fixture
def identity_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    """Laya selected: its routing (identity mapping, its own threshold) comes from config."""
    monkeypatch.setattr(get_settings().llm.intent_classifier, "provider", "laya_local")


@pytest.mark.asyncio
@pytest.mark.usefixtures("identity_mapping")
async def test_an_intent_the_bert_labels_cannot_reach_is_routed() -> None:
    router = CustomerSupportIntentRouter(backend=IntentNamedBackend("LIST_ORDERS", 0.9))

    result = await router.classify_async("anything I bought that hasn't shipped?")

    assert result.name == "LIST_ORDERS"
    assert result.allowed_tools == ["list_orders"]


@pytest.mark.asyncio
@pytest.mark.usefixtures("identity_mapping")
async def test_the_source_names_the_configured_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(get_settings().llm.intent_classifier, "provider", "laya_local")
    router = CustomerSupportIntentRouter(backend=IntentNamedBackend("ORDER_DETAILS", 0.9))

    result = await router.classify_async("what did order 10001 contain?")

    assert result.source == "laya_local"


def test_the_default_backend_is_given_the_agents_intents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: dict[str, Any] = {}

    def factory(**kwargs: Any) -> IntentNamedBackend:
        received.update(kwargs)
        return IntentNamedBackend("UNKNOWN", 1.0)

    monkeypatch.setattr(router_module, "create_intent_classifier", factory)
    settings = get_agent_settings()

    CustomerSupportIntentRouter()

    assert received["choices"] == {
        **{intent.name: intent.description for intent in settings.intents},
    }
    assert received["instructions"] == settings.intent_router.instructions


class CountingBackend(IntentNamedBackend):
    def __init__(self, top: str, score: float) -> None:
        super().__init__(top, score)
        self.calls = 0

    async def classify(self, text: str) -> list[IntentCandidate]:
        self.calls += 1
        return await super().classify(text)


@pytest.mark.asyncio
@pytest.mark.usefixtures("identity_mapping")
async def test_with_phrase_rules_off_a_phrase_message_is_left_to_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(get_agent_settings().intent_router, "phrase_rules", False)
    backend = CountingBackend("ORDER_STATUS", 0.9)
    router = CustomerSupportIntentRouter(backend=backend)

    result = await router.classify_async("cancel order 10622")

    assert backend.calls == 1
    assert result.name == "ORDER_STATUS"
    assert result.source != "deterministic"


@pytest.mark.asyncio
@pytest.mark.usefixtures("identity_mapping")
async def test_with_no_reject_rules_an_off_topic_message_reaches_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(get_agent_settings().intent_router, "deterministic_rejects", [])
    backend = CountingBackend("UNKNOWN", 0.9)
    router = CustomerSupportIntentRouter(backend=backend)

    await router.classify_async("tell me the weather")

    assert backend.calls == 1


@pytest.mark.asyncio
async def test_phrase_rules_are_on_by_default() -> None:
    backend = CountingBackend("ORDER_STATUS", 0.9)
    router = CustomerSupportIntentRouter(backend=backend)

    result = await router.classify_async("cancel order 10622")

    assert get_agent_settings().intent_router.phrase_rules is True
    assert result.name == "CANCEL_ORDER"
    assert backend.calls == 0


class TestTheClassifierIsChosenInConfig:
    def test_laya_with_phrase_rules_is_the_default(self) -> None:
        router = get_agent_settings().intent_router

        assert get_settings().llm.intent_classifier.provider == "laya_local"
        assert router.phrase_rules is True
        assert router.routing("laya_local").confidence_threshold == 0.6

    def test_each_classifier_brings_its_own_threshold_and_labels(self) -> None:
        router = get_agent_settings().intent_router
        bert = router.routing("huggingface_local")
        laya = router.routing("laya_local")

        assert bert.confidence_threshold == 0.75
        assert "track_order" in bert.model_label_mapping["ORDER_STATUS"]
        assert laya.model_label_mapping["LIST_ORDERS"] == ["LIST_ORDERS"]

    def test_a_classifier_with_no_routing_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no routing"):
            get_agent_settings().intent_router.routing("unheard_of")

    @pytest.mark.asyncio
    async def test_bert_labels_route_when_bert_is_selected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(get_settings().llm.intent_classifier, "provider", "huggingface_local")
        router = CustomerSupportIntentRouter(backend=IntentNamedBackend("track_order", 0.8))

        result = await router.classify_async("is my parcel 10943 still with the courier?")

        assert result.name == "ORDER_STATUS"

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("identity_mapping")
    async def test_laya_routes_at_its_own_threshold(self) -> None:
        router = CustomerSupportIntentRouter(backend=IntentNamedBackend("ORDER_DETAILS", 0.62))

        result = await router.classify_async("what did order 10001 contain?")

        assert result.name == "ORDER_DETAILS"
