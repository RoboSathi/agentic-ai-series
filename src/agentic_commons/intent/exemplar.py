"""Intent classification by nearest labelled example: no training, a handful of phrases per label.

Each label comes with example messages; a new message is embedded and compared with every example,
and the labels of its nearest few vote, weighted by similarity. The votes, normalised, are the
scores — the same labels-and-scores shape a fine-tuned classifier gives, so the same confidence
tests apply. The examples are the agent's own data and the embedder is injected; nothing here
knows what a label means.

It beat zero-shot NLI on the research agent's own examples (91% against 80% leave-one-out) because
"a quick lookup or a deep report" is a property of how a request is phrased, not of its topic.
"""

import asyncio
from collections import defaultdict
from collections.abc import Mapping, Sequence

from agentic_commons.intent.types import IntentCandidate
from agentic_commons.llm.embeddings import EmbeddingClient, cosine_similarity


class ExemplarIntentClassifier:
    """Embeds the examples once (on first use), then votes among a message's nearest ones."""

    def __init__(
        self,
        embedder: EmbeddingClient,
        exemplars: Mapping[str, Sequence[str]],
        neighbours: int,
    ) -> None:
        pairs = [(label, text) for label, texts in exemplars.items() for text in texts]
        if not pairs or neighbours < 1:
            raise ValueError("an exemplar classifier needs examples and at least one neighbour")
        self._embedder = embedder
        self._pairs = pairs
        self._labels = list(exemplars)
        self._neighbours = neighbours
        self._vectors: list[list[float]] | None = None
        self._indexing = asyncio.Lock()

    @property
    def model_name(self) -> str:
        """The embedding model whose geometry decides the nearest examples."""
        return self._embedder.model_id

    async def classify(self, text: str) -> list[IntentCandidate]:
        """Every label with its share of the nearest examples' votes, best first."""
        vectors = await self._index()
        query = (await self._embedder.embed([text]))[0]
        if len(query) != len(vectors[0]):
            raise RuntimeError("the embedder changed since the examples were embedded")
        nearest = sorted(
            (
                (cosine_similarity(query, vector), label)
                for vector, (label, _) in zip(vectors, self._pairs)
            ),
            reverse=True,
        )[: self._neighbours]
        votes: dict[str, float] = defaultdict(float)
        for similarity, label in nearest:
            votes[label] += max(similarity, 0.0)
        total = sum(votes.values()) or 1.0
        ranked = sorted(self._labels, key=lambda label: votes[label], reverse=True)
        return [IntentCandidate(intent=label, score=votes[label] / total) for label in ranked]

    async def _index(self) -> list[list[float]]:
        """The examples' vectors, computed once, however many messages ask at the same time."""
        async with self._indexing:
            if self._vectors is None:
                self._vectors = await self._embedder.embed([text for _, text in self._pairs])
            return self._vectors
