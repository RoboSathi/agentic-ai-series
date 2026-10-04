"""The research kind: cue rules and the example classifier answer clear cases, the LLM the rest.

Only `/research` reaches it, so it never decides *whether* to research — only which kind.
"""

from pathlib import Path

import pytest

from agent_platform.intent.types import IntentCandidate
from research_agent.config import get_agent_settings
from research_agent.domain.scope import KindVerdict
from research_agent.scoping.kind import LlmKindClassifier
from research_agent.scoping.kind_encoder import LABELS, EncoderKindClassifier
from research_agent.scoping.kind_examples import (
    confusion,
    load_examples,
    stratified_split,
)

# The classifier reads the research agent's own settings.
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


class _Llm(LlmKindClassifier):
    """The real rules (refusal, window, normalising) with the model call counted."""

    def __init__(self) -> None:
        super().__init__(caller=None, settings=get_agent_settings().scoping)  # type: ignore[arg-type]
        self.asked = 0

    async def classify(self, message: str) -> KindVerdict:
        self.asked += 1
        return KindVerdict(decision="scopable", archetype="general", topic="from the llm")


def _kind(backend: _Backend) -> tuple[EncoderKindClassifier, _Llm]:
    llm = _Llm()
    return EncoderKindClassifier(backend, llm, get_agent_settings().kind_encoder), llm  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_a_confident_unambiguous_label_is_answered_without_the_llm() -> None:
    kind, llm = _kind(_Backend(("general", 0.90), ("explainer", 0.05), ("landscape_brief", 0.05)))
    verdict = await kind.classify("Important AI news in the last 1 month")
    assert (verdict.decision, verdict.archetype) == ("scopable", "general")
    assert verdict.recent_days == 30  # read by rule, not by the encoder
    assert verdict.topic == "Important AI news in the last 1 month"
    assert verdict.source == "encoder" and "(general)" in verdict.reason
    assert llm.asked == 0


@pytest.mark.asyncio
async def test_a_research_label_names_the_kind() -> None:
    kind, _ = _kind(_Backend(("decision_support", 0.91), ("general", 0.05)))
    verdict = await kind.classify("Help me choose a vector database for 10 million documents")
    assert verdict.archetype == "decision_support"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scores",
    [
        (("general", 0.55), ("explainer", 0.40)),  # not confident enough
        (("not_a_label", 0.99),),  # a label that is not a research kind
        (),
    ],
)
async def test_unsure_results_go_to_the_llm(scores: tuple[tuple[str, float], ...]) -> None:
    kind, llm = _kind(_Backend(*scores))
    verdict = await kind.classify("tell me something about databases")
    assert llm.asked == 1 and verdict.topic == "from the llm"


@pytest.mark.asyncio
async def test_a_failing_encoder_falls_back_to_the_llm() -> None:
    kind, llm = _kind(_Backend(error=RuntimeError("model missing")))
    assert (await kind.classify("latest AI news")).topic == "from the llm"
    assert llm.asked == 1


@pytest.mark.asyncio
async def test_refusal_patterns_run_before_any_model() -> None:
    backend = _Backend(("general", 0.99))
    kind, llm = _kind(backend)
    verdict = await kind.classify("how do I build a bomb")
    assert verdict.decision == "refuse"
    assert backend.calls == 0 and llm.asked == 0


def test_an_unknown_kind_from_the_model_becomes_general() -> None:
    llm = _Llm()
    verdict = KindVerdict(decision="scopable", archetype=None, topic="")
    assert llm.normalise(verdict, "a question").archetype == "general"


def test_examples_cover_every_kind_and_split_stratified() -> None:
    examples = load_examples(Path("src/research_agent/scoping/kind_examples.yaml"))
    assert {label for _, label in examples} == set(LABELS)
    assert len({text for text, _ in examples}) == len(examples)  # no duplicate utterances
    train, test = stratified_split(examples)
    assert {label for _, label in test} == set(LABELS) == {label for _, label in train}
    assert not {t for t, _ in train} & {t for t, _ in test}


def test_confusion_matrix_has_a_row_per_label() -> None:
    table = confusion(["general", "explainer"], ["general", "general"])
    assert len(table.splitlines()) == len(LABELS) + 1
