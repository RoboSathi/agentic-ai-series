"""The Laya intent provider: one choice question built from injected labels, nothing downloaded."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from agent_platform.config import get_settings
from agent_platform.config.settings import IntentClassifierSettings
from agent_platform.intent import IntentClassifierInterface, create_intent_classifier
from agent_platform.intent.laya import LayaIntentClassifier
from agent_platform.llm import local_laya

CHOICES = {
    "ORDER_STATUS": "where an order is",
    "REQUEST_REFUND": "asking for money back",
    "UNKNOWN": "anything else",
}
INSTRUCTIONS = "Which request is the customer making?"


class FakeAgent:
    """Answers every question with fixed probabilities and remembers what it was asked."""

    def __init__(self, probabilities: dict[str, float]) -> None:
        self._probabilities = probabilities
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def system_one(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((state, questions))
        return {
            "answers": {
                qid: {"type": "choice", "probabilities": dict(self._probabilities)}
                for qid in questions
            }
        }


def _settings(tmp_path: Path, **overrides: Any) -> IntentClassifierSettings:
    base = get_settings().llm.intent_classifier
    laya = base.laya.model_copy(
        update={
            "model": "org/laya",
            "subfolder": "typed-decisions",
            "local_model_dir": str(tmp_path / "laya"),
            "question_id": "intent",
            "state_key": "message",
        }
    )
    return base.model_copy(
        update={"enabled": True, "provider": "laya_local", "laya": laya, **overrides}
    )


def _artifact(directory: Path) -> None:
    directory.mkdir(parents=True)
    for name in ("model.safetensors", "rl_agent_config.json"):
        (directory / name).write_text("x")


def _classifier(
    tmp_path: Path, agent: FakeAgent, fetched: list[Path] | None = None, **overrides: Any
) -> LayaIntentClassifier:
    def fetch(repo: str, subfolder: str, destination: Path) -> None:
        if fetched is not None:
            fetched.append(destination)
        _artifact(destination)

    return LayaIntentClassifier(
        CHOICES,
        INSTRUCTIONS,
        settings=_settings(tmp_path, **overrides),
        loader=lambda path, device: agent,
        fetcher=fetch,
    )


@pytest.mark.asyncio
async def test_asks_one_choice_question_built_from_the_injected_labels(tmp_path: Path) -> None:
    agent = FakeAgent({"ORDER_STATUS": 0.2, "REQUEST_REFUND": 0.7, "UNKNOWN": 0.1})
    _artifact(tmp_path / "laya")
    classifier = _classifier(tmp_path, agent)

    await classifier.classify("money back for 10942")

    state, questions = agent.calls[0]
    assert state == {"message": "money back for 10942"}
    assert questions == {
        "intent": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": CHOICES}
    }


@pytest.mark.asyncio
async def test_returns_every_label_best_first(tmp_path: Path) -> None:
    agent = FakeAgent({"ORDER_STATUS": 0.2, "REQUEST_REFUND": 0.7, "UNKNOWN": 0.1})
    _artifact(tmp_path / "laya")

    candidates = await _classifier(tmp_path, agent).classify("money back")

    assert [(c.intent, c.score) for c in candidates] == [
        ("REQUEST_REFUND", 0.7),
        ("ORDER_STATUS", 0.2),
        ("UNKNOWN", 0.1),
    ]


def test_satisfies_the_platform_intent_contract(tmp_path: Path) -> None:
    classifier = _classifier(tmp_path, FakeAgent({}))

    assert isinstance(classifier, IntentClassifierInterface)
    assert classifier.model_name == "org/laya/typed-decisions"
    assert classifier.score_type == "laya_calibrated"
    assert classifier.cache_path == tmp_path / "laya"
    assert classifier.is_loaded is False


def test_loads_from_the_local_folder_without_downloading(tmp_path: Path) -> None:
    fetched: list[Path] = []
    _artifact(tmp_path / "laya")
    classifier = _classifier(tmp_path, FakeAgent({}), fetched)

    classifier.ensure_loaded_sync()

    assert classifier.is_loaded is True
    assert fetched == []


def test_downloads_once_when_the_folder_is_missing(tmp_path: Path) -> None:
    fetched: list[Path] = []
    classifier = _classifier(tmp_path, FakeAgent({}), fetched)

    classifier.ensure_loaded_sync()
    classifier.ensure_loaded_sync()

    assert len(fetched) == 1
    assert (tmp_path / "laya" / "model.safetensors").exists()


def test_refuses_a_missing_artifact_when_downloads_are_off(tmp_path: Path) -> None:
    classifier = _classifier(tmp_path, FakeAgent({}), download_if_missing=False)

    with pytest.raises(RuntimeError, match="unavailable"):
        classifier.ensure_loaded_sync()


@pytest.mark.asyncio
async def test_a_disabled_classifier_refuses_to_classify(tmp_path: Path) -> None:
    classifier = _classifier(tmp_path, FakeAgent({}), enabled=False)

    with pytest.raises(RuntimeError, match="disabled"):
        await classifier.classify("refund")


def test_says_how_to_install_laya_when_it_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "laya", None)
    local_laya.resident_laya.cache_clear()
    _artifact(tmp_path / "laya")
    classifier = LayaIntentClassifier(CHOICES, INSTRUCTIONS, settings=_settings(tmp_path))

    with pytest.raises(RuntimeError, match="not installed"):
        classifier.ensure_loaded_sync()


def test_needs_at_least_two_labels(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="two"):
        LayaIntentClassifier({"ONLY": "one"}, INSTRUCTIONS, settings=_settings(tmp_path))


def test_factory_builds_laya_only_with_the_agents_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = get_settings().llm
    monkeypatch.setattr(llm, "intent_classifier", _settings(tmp_path))

    with pytest.raises(ValueError, match="labels"):
        create_intent_classifier()
    built = create_intent_classifier(choices=CHOICES, instructions=INSTRUCTIONS)

    assert isinstance(built, LayaIntentClassifier)


def test_one_checkpoint_is_loaded_once_per_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loads: list[str] = []
    fake = type(sys)("laya")
    fake.load = lambda path, device: loads.append(path) or FakeAgent({})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "laya", fake)
    local_laya.resident_laya.cache_clear()
    _artifact(tmp_path / "laya")

    for _ in range(2):
        LayaIntentClassifier(
            CHOICES, INSTRUCTIONS, settings=_settings(tmp_path)
        ).ensure_loaded_sync()

    assert loads == [str(tmp_path / "laya")]
    local_laya.resident_laya.cache_clear()


def test_model_check_probes_the_classifier_in_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_platform.llm import local_probes

    config = get_settings()
    monkeypatch.setattr(config.llm, "intent_classifier", _settings(tmp_path))

    probe = local_probes._intent_probe(config)

    assert probe is not None
    assert probe.name == "intent_classifier"
    assert probe._model == "org/laya/typed-decisions"
    assert probe._path == str(tmp_path / "laya")


class _Logits:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def __getitem__(self, index: int) -> "_Logits":
        return self


class _Torch:
    """Just enough of torch for the BERT provider: no_grad and softmax over a list."""

    class no_grad:  # noqa: N801 - mirrors torch.no_grad
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_: object) -> None:
            return None

    @staticmethod
    def softmax(logits: _Logits, dim: int) -> Any:
        import math

        exps = [math.exp(v) for v in logits.values]
        total = sum(exps)
        return type("Probs", (), {"tolist": lambda self: [e / total for e in exps]})()


async def test_the_bert_provider_answers_in_its_own_labels_best_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_platform.intent.classifier import HuggingFaceIntentClassifier

    model = type(
        "Model",
        (),
        {
            "config": type("Config", (), {"id2label": {0: "track_order", 1: "get_refund"}})(),
            "__call__": lambda self, **_: type("Out", (), {"logits": _Logits([0.0, 2.0])})(),
        },
    )()

    async def loaded(self: Any) -> tuple[Any, Any, Any]:
        return (lambda text, **_: {}), model, _Torch

    monkeypatch.setattr("agent_platform.llm.local_models.LocalSequenceClassifier.load", loaded)
    monkeypatch.setattr(get_settings().llm.intent_classifier, "enabled", True)

    candidates = await HuggingFaceIntentClassifier().classify("money back for 10942")

    assert [c.intent for c in candidates] == ["get_refund", "track_order"]
    assert candidates[0].score > 0.85


def test_the_factory_builds_bert_when_it_is_selected(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent_platform.intent.classifier import HuggingFaceIntentClassifier

    monkeypatch.setattr(get_settings().llm.intent_classifier, "provider", "huggingface_local")

    assert isinstance(create_intent_classifier(choices=CHOICES), HuggingFaceIntentClassifier)
