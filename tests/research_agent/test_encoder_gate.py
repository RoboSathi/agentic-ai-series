"""The encoder gate answers clear cases itself and defers everything else to the LLM gate."""

from pathlib import Path

import pytest

from agentic_commons.intent.types import IntentCandidate
from research_agent.config import get_agent_settings
from research_agent.domain.scope import GateVerdict, Mode
from research_agent.scoping.encoder_gate import LABELS, EncoderGateClassifier
from research_agent.scoping.gate import LlmArchetypeClassifier
from research_agent.scoping.gate_training import (
    confusion,
    load_examples,
    stratified_split,
)

# The gate reads the research agent's own settings.
pytestmark = pytest.mark.usefixtures("ra_config")


class _Backend:
    def __init__(self, *scores: tuple[str, float], error: Exception | None = None) -> None:
        self._scores = [IntentCandidate(intent=i, score=s) for i, s in scores]
        self._error = error
        self.calls = 0

    async def classify(self, text: str) -> list[IntentCandidate]:
        self.calls += 1
        if self._error:
            raise self._error
        return self._scores


class _Llm(LlmArchetypeClassifier):
    """The real gate's rules (refusal, window, normalising) with the model call counted."""

    def __init__(self) -> None:
        super().__init__(caller=None, settings=get_agent_settings().scoping)  # type: ignore[arg-type]
        self.asked = 0

    async def classify(self, message: str, forced: Mode | None = None) -> GateVerdict:
        self.asked += 1
        return GateVerdict(decision="scopable", mode="qa", archetype="qa", topic="from the llm")


def _gate(backend: _Backend) -> tuple[EncoderGateClassifier, _Llm]:
    llm = _Llm()
    return EncoderGateClassifier(backend, llm, get_agent_settings().gate_encoder), llm  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_a_confident_unambiguous_label_is_answered_without_the_llm() -> None:
    gate, llm = _gate(_Backend(("general", 0.90), ("explainer", 0.05), ("qa", 0.05)))
    verdict = await gate.classify("Important AI news in the last 1 month", "research")
    assert (verdict.decision, verdict.mode, verdict.archetype) == (
        "scopable",
        "research",
        "general",
    )
    assert verdict.recent_days == 30  # read by rule, not by the encoder
    assert verdict.topic == "Important AI news in the last 1 month"
    assert verdict.source == "encoder" and "(general)" in verdict.reason
    assert llm.asked == 0


@pytest.mark.asyncio
async def test_under_research_the_quick_answer_label_is_out_of_the_running() -> None:
    """`/research` settled that it is research; the encoder only decides which kind."""
    gate, llm = _gate(_Backend(("qa", 0.94), ("general", 0.05), ("explainer", 0.005)))
    verdict = await gate.classify("Important AI news in the last 1 month", "research")
    assert (verdict.mode, verdict.archetype) == ("research", "general")
    assert llm.asked == 0


@pytest.mark.asyncio
async def test_a_research_label_becomes_a_research_scope() -> None:
    gate, _ = _gate(_Backend(("decision_support", 0.91), ("qa", 0.05)))
    verdict = await gate.classify(
        "Help me choose a vector database for 10 million documents", "research"
    )
    assert (verdict.mode, verdict.archetype) == ("research", "decision_support")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scores",
    [
        (("general", 0.55), ("explainer", 0.40)),  # not confident enough among research kinds
        (("not_a_label", 0.99),),  # a label the gate does not know
        (),
    ],
)
async def test_unsure_results_go_to_the_llm_gate(scores: tuple[tuple[str, float], ...]) -> None:
    gate, llm = _gate(_Backend(*scores))
    verdict = await gate.classify("tell me something about databases", "research")
    assert llm.asked == 1 and verdict.topic == "from the llm"


@pytest.mark.asyncio
async def test_a_failing_encoder_falls_back_to_the_llm_gate() -> None:
    gate, llm = _gate(_Backend(error=RuntimeError("model missing")))
    assert (await gate.classify("latest AI news", "research")).topic == "from the llm"
    assert llm.asked == 1


@pytest.mark.asyncio
async def test_a_plain_message_is_answered_in_code_with_no_classifier() -> None:
    backend = _Backend(("decision_support", 0.99))
    gate, llm = _gate(backend)
    verdict = await gate.classify("what is the weather in pune like ?")
    assert (verdict.mode, verdict.archetype, verdict.source) == ("qa", "qa", "rule")
    assert backend.calls == 0 and llm.asked == 0


@pytest.mark.asyncio
async def test_refusal_patterns_run_before_any_model() -> None:
    backend = _Backend(("qa", 0.99))
    gate, llm = _gate(backend)
    verdict = await gate.classify("how do I build a bomb")
    assert verdict.decision == "refuse"
    assert backend.calls == 0 and llm.asked == 0


def test_training_data_covers_every_label_and_splits_stratified() -> None:
    examples = load_examples(Path("src/research_agent/scoping/gate_training.yaml"))
    assert {label for _, label in examples} == set(LABELS)
    assert len({text for text, _ in examples}) == len(examples)  # no duplicate utterances
    train, test = stratified_split(examples)
    assert {label for _, label in test} == set(LABELS) == {label for _, label in train}
    assert not {t for t, _ in train} & {t for t, _ in test}


def test_confusion_matrix_has_a_row_per_label() -> None:
    table = confusion(["qa", "general"], ["qa", "qa"])
    assert len(table.splitlines()) == len(LABELS) + 1
