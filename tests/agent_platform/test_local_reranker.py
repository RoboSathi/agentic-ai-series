"""The reranker runs a local cross-encoder, and degrades to its fallback when it cannot."""

from types import SimpleNamespace
from typing import Any

import pytest

from agent_platform.config import get_settings
from agent_platform.llm.local_reranker import LocalCrossEncoderRanker, create_reranker
from tests.conftest import OfflineArtifact

torch = pytest.importorskip("torch")


class _Tokenizer:
    """Encodes each candidate as its length, so a longer candidate scores higher."""

    def __call__(self, queries: list[str], candidates: list[str], **_: Any) -> dict[str, Any]:
        assert len(queries) == len(candidates)
        return {"input_ids": torch.tensor([[float(len(text))] for text in candidates])}


class _Model:
    """One relevance logit per pair, like ms-marco."""

    def __call__(self, input_ids: Any) -> Any:
        return SimpleNamespace(logits=input_ids - 5.0)


class _LoadedArtifact(OfflineArtifact):
    def load_sync(self) -> Any:
        return _Tokenizer(), _Model(), torch


class _Fallback:
    def __init__(self) -> None:
        self.calls = 0

    async def rank(self, text: str, candidates: list[str]) -> list[float]:
        self.calls += 1
        return [float(index) for index, _ in enumerate(candidates)]


async def test_each_candidate_is_scored_in_order_as_a_probability() -> None:
    ranker = LocalCrossEncoderRanker(artifact=_LoadedArtifact())  # type: ignore[arg-type]
    scores = await ranker.rank("where is my refund", ["abc", "abcdefghij", "a"])
    assert len(scores) == 3
    assert all(0.0 < score < 1.0 for score in scores)
    assert scores[1] > scores[0] > scores[2]


async def test_an_empty_shortlist_scores_nothing() -> None:
    ranker = LocalCrossEncoderRanker(artifact=_LoadedArtifact())  # type: ignore[arg-type]
    assert await ranker.rank("query", []) == []


async def test_a_model_that_cannot_load_latches_to_the_fallback() -> None:
    fallback = _Fallback()
    ranker = LocalCrossEncoderRanker(artifact=OfflineArtifact(), fallback=fallback)  # type: ignore[arg-type]
    assert await ranker.rank("query", ["a", "b"]) == [0.0, 1.0]
    assert await ranker.rank("query", ["a", "b"]) == [0.0, 1.0]
    assert fallback.calls == 2


async def test_with_no_fallback_an_unavailable_ranker_leaves_the_order_alone() -> None:
    ranker = LocalCrossEncoderRanker(artifact=OfflineArtifact())  # type: ignore[arg-type]
    scores = await ranker.rank("query", ["first", "second", "third"])
    assert len(set(scores)) == 1


def test_the_factory_builds_nothing_with_encoders_off(restore_settings: None) -> None:
    get_settings().llm.encoders.enabled = False
    assert create_reranker() is None


def test_the_factory_builds_a_local_ranker() -> None:
    assert isinstance(create_reranker(), LocalCrossEncoderRanker)
