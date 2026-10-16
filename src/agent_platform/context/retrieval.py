"""Production retrieval — lexical, dense, hybrid, graph (LLD §7).

The decision rule, implemented rather than described:

  identifier-shaped query  → lexical wins  (BM25 finds "ORD-10042"; an
                             embedding of an order number means nothing)
  relationship-shaped      → graph wins    (hops, not similarity)
  narrative / paraphrased  → dense wins    (wording differs, meaning doesn't)

Hybrid exists because real queries are mixtures, and picking one mode
globally means being wrong for whichever half of the traffic doesn't fit.
"""

import heapq
import math
import re
from collections import Counter, defaultdict
from typing import Protocol, runtime_checkable

from agent_platform.config import get_settings
from agent_platform.context.types import Chunk
from agent_platform.llm.embeddings import tokenize
from agent_platform.logging import get_logger

# A query is "identifier-shaped" if it contains a token that looks like a
# code, id or reference rather than a word.
_IDENTIFIER = re.compile(r"\b(?=[\w-]*\d)[A-Za-z0-9][\w-]{3,}\b")


@runtime_checkable
class Retriever(Protocol):
    """Finds the k most useful chunks for a query."""

    async def retrieve(self, query: str, k: int) -> list[Chunk]: ...


class LexicalRetriever:
    """BM25 over an in-memory corpus, scored through an inverted index.

    BM25 rather than raw TF-IDF because the two parameters matter: `k1`
    caps how much repeating a term can help, and `b` controls how much a
    long document is penalized for its length. Both are in config.

    `index()` is one pass: postings (term → document → frequency) and each
    document's length norm. A query then touches only the documents that
    share one of its terms, its IDFs are computed for its own terms alone,
    and only the k winners are copied. Several callers index per claim, so
    the build has to stay as cheap as the search.
    """

    def __init__(self, corpus: list[Chunk] | None = None) -> None:
        settings = get_settings().context
        self._k1 = settings.bm25_k1
        self._b = settings.bm25_b
        self._corpus: list[Chunk] = []
        self._postings: dict[str, dict[int, int]] = {}
        self._norms: list[float] = []
        if corpus:
            self.index(corpus)

    def index(self, chunks: list[Chunk]) -> None:
        """Build the postings and length norms BM25 needs."""
        self._corpus = list(chunks)
        self._postings = {}
        lengths: list[int] = []
        for doc_index, chunk in enumerate(self._corpus):
            tokens = tokenize(chunk.content)
            lengths.append(len(tokens))
            for term, freq in Counter(tokens).items():
                self._postings.setdefault(term, {})[doc_index] = freq
        average = sum(lengths) / len(lengths) if lengths else 0.0
        self._norms = [
            1 - self._b + self._b * (length / average if average else 1.0) for length in lengths
        ]

    async def retrieve(self, query: str, k: int) -> list[Chunk]:
        """Top-k by BM25 score; a document sharing no query term is never scored."""
        scores = self._scores(query)
        best = heapq.nlargest(k, scores.items(), key=lambda item: (item[1], -item[0]))
        return [
            self._corpus[doc_index].model_copy(update={"score": score})
            for doc_index, score in best
            if score > 0
        ]

    def score(self, query: str, doc_index: int) -> float:
        """BM25 score of one document against the query."""
        return self._scores(query, only=doc_index).get(doc_index, 0.0)

    def _scores(self, query: str, only: int | None = None) -> dict[int, float]:
        """Every matching document's score, summed term by term over the postings."""
        scores: dict[int, float] = defaultdict(float)
        for term in tokenize(query):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf(len(postings))
            matched = postings.items() if only is None else ((only, postings.get(only, 0)),)
            for doc_index, freq in matched:
                if freq:
                    scores[doc_index] += self._term_score(idf, freq, self._norms[doc_index])
        return scores

    def _idf(self, doc_freq: int) -> float:
        total = len(self._corpus)
        return math.log(1 + (total - doc_freq + 0.5) / (doc_freq + 0.5))

    def _term_score(self, idf: float, freq: int, norm: float) -> float:
        return idf * (freq * (self._k1 + 1)) / (freq + self._k1 * norm)


class DenseRetriever:
    """Embedding similarity over the same corpus, via the injected ranker."""

    def __init__(self, ranker: "RelevanceRanker", corpus: list[Chunk] | None = None) -> None:
        self._ranker = ranker
        self._corpus = list(corpus or [])

    def index(self, chunks: list[Chunk]) -> None:
        """Replace the corpus this retriever searches."""
        self._corpus = list(chunks)

    async def retrieve(self, query: str, k: int) -> list[Chunk]:
        """Top-k by embedding similarity."""
        if not self._corpus:
            return []
        scores = await self._ranker.rank(query, [chunk.content for chunk in self._corpus])
        scored = [
            chunk.model_copy(update={"score": score}) for chunk, score in zip(self._corpus, scores)
        ]
        return sorted(scored, key=lambda chunk: chunk.score, reverse=True)[:k]


@runtime_checkable
class RelevanceRanker(Protocol):
    """Scores candidates against a query (the NOLLM cascade's `rank`)."""

    async def rank(self, text: str, candidates: list[str]) -> list[float]: ...


class GraphRetriever:
    """Relationship traversal over an explicit edge set (GraphRAG, optional).

    Included because some questions are hops, not similarity: "which
    orders share the payment method that was charged back?" is answerable
    by walking edges and hopeless by cosine distance.
    """

    def __init__(self, edges: dict[str, list[str]] | None = None) -> None:
        self._edges = edges or {}
        self._nodes: dict[str, Chunk] = {}

    def index(self, chunks: list[Chunk], edges: dict[str, list[str]]) -> None:
        """Load the node contents and the edges between them."""
        self._nodes = {chunk.source: chunk for chunk in chunks}
        self._edges = edges

    async def retrieve(self, query: str, k: int) -> list[Chunk]:
        """Breadth-first from any node the query names, out to k results."""
        seeds = [node for node in self._nodes if node and node.casefold() in query.casefold()]
        visited: list[str] = []
        frontier = list(seeds)
        while frontier and len(visited) < k:
            node = frontier.pop(0)
            if node in visited:
                continue
            visited.append(node)
            frontier.extend(self._edges.get(node, []))
        return [self._nodes[node] for node in visited if node in self._nodes][:k]


class HybridRetriever:
    """Lexical + dense, fused by a configured weight (Composite).

    Fusion is on normalized scores: BM25 is unbounded and cosine is in
    [0, 1], so combining the raw numbers would let one mode dominate for
    reasons that have nothing to do with relevance.
    """

    def __init__(
        self,
        lexical: LexicalRetriever | None = None,
        dense: DenseRetriever | None = None,
        graph: GraphRetriever | None = None,
    ) -> None:
        settings = get_settings().context
        self._lexical = lexical or LexicalRetriever()
        self._dense = dense
        self._graph = graph
        self._mode = settings.retrieval_mode
        self._lexical_weight = settings.hybrid_lexical_weight
        self._logger = get_logger("context")

    def index(self, chunks: list[Chunk]) -> None:
        """Index the corpus into every configured sub-retriever."""
        self._lexical.index(chunks)
        if self._dense is not None:
            self._dense.index(chunks)

    async def retrieve(self, query: str, k: int) -> list[Chunk]:
        """Retrieve by the configured mode, or by the query's shape in hybrid.

        A mode whose retriever was not given falls back to the next one that was, and says
        so: `graph` with no edges declared is lexical/hybrid retrieval, not graph retrieval.
        In hybrid, a declared graph adds the neighbours of the fused hits (one hop).
        """
        if self._mode == "graph" and self._graph is not None:
            return await self._graph.retrieve(query, k)
        if self._mode in ("graph", "dense") and (self._mode == "graph" or self._dense is None):
            self._logger.debug("Retrieval mode not available; falling back", mode=self._mode)
        if self._mode == "dense" and self._dense is not None:
            return await self._dense.retrieve(query, k)
        if self._mode == "lexical" or self._dense is None:
            return await self._lexical.retrieve(query, k)
        fused = await self._fuse(query, k)
        return await self._expanded(query, fused, k)

    async def _expanded(self, query: str, fused: list[Chunk], k: int) -> list[Chunk]:
        """The fused hits plus their graph neighbours, up to k (unchanged without a graph)."""
        if self._graph is None or len(fused) >= k:
            return fused
        seen = {chunk.source for chunk in fused}
        hops = await self._graph.retrieve(" ".join([query, *seen]), k)
        return [*fused, *(chunk for chunk in hops if chunk.source not in seen)][:k]

    async def _fuse(self, query: str, k: int) -> list[Chunk]:
        """Weighted fusion, tilted toward lexical for identifier queries."""
        assert self._dense is not None
        weight = 1.0 if is_identifier_query(query) else self._lexical_weight
        lexical = _normalize(await self._lexical.retrieve(query, k * 2))
        dense = _normalize(await self._dense.retrieve(query, k * 2))
        combined: dict[str, Chunk] = {}
        for chunk in lexical:
            combined[chunk.content] = chunk.model_copy(update={"score": weight * chunk.score})
        for chunk in dense:
            existing = combined.get(chunk.content)
            blended = (1 - weight) * chunk.score + (existing.score if existing else 0.0)
            combined[chunk.content] = chunk.model_copy(update={"score": blended})
        return sorted(combined.values(), key=lambda chunk: chunk.score, reverse=True)[:k]


class CrossEncoderReranker:
    """Reorders a shortlist with a stronger, slower scorer.

    Worth its cost only over a shortlist: reranking is quadratic in
    attention over the pair, so it runs on the ten candidates retrieval
    surfaced, never over the corpus.
    """

    def __init__(self, ranker: RelevanceRanker) -> None:
        self._ranker = ranker

    async def rerank(self, query: str, chunks: list[Chunk], k: int) -> list[Chunk]:
        """Rescore a shortlist and return the best k."""
        if not chunks:
            return []
        scores = await self._ranker.rank(query, [chunk.content for chunk in chunks])
        rescored = [
            chunk.model_copy(update={"score": score}) for chunk, score in zip(chunks, scores)
        ]
        return sorted(rescored, key=lambda chunk: chunk.score, reverse=True)[:k]


def is_identifier_query(query: str) -> bool:
    """Whether a query is shaped like an identifier lookup."""
    return bool(_IDENTIFIER.search(query))


def _normalize(chunks: list[Chunk]) -> list[Chunk]:
    """Scale scores into [0, 1] so two retrievers can be compared at all."""
    if not chunks:
        return []
    top = max(chunk.score for chunk in chunks)
    if top <= 0:
        return chunks
    return [chunk.model_copy(update={"score": chunk.score / top}) for chunk in chunks]
