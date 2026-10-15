"""The five-stage context assembly pipeline (LLD §7).

    gather_candidates → score_relevance → fit_to_budget → format → order

Each stage is separately testable and does one thing. The ordering is
load-bearing: scoring before fitting means the budget cuts the least
relevant material rather than whatever happened to arrive last, and
ordering *after* fitting means position is decided over the material
that actually survived.

Fitting is two steps. The strategies go first (`WindowFitter`): they say a
repeated thing once, keep the task's kinds, hold the stable chunks fixed and,
while the window is still over budget, retrieve, offload and reduce. Only what
still does not fit after them is cut — so a strategy is what decides what the
model loses, not the order chunks happened to be scored in.
"""

from typing import Protocol, runtime_checkable

from agent_platform.config import get_settings
from agent_platform.context.retrieval import CrossEncoderReranker
from agent_platform.context.similarity import jaccard
from agent_platform.context.trust import with_trust
from agent_platform.context.types import (
    STABLE,
    AssembledContext,
    Chunk,
    ChunkKind,
    Task,
    provenance_header,
)
from agent_platform.llm.tokenizer import count_tokens
from agent_platform.logging import get_logger, instrument, logged

# Where each kind of content belongs in the window. Instructions lead;
# retrieved material sits in the middle; the task is last so it is the
# most recent thing the model reads (the "lost in the middle" ordering).
_KIND_ORDER: dict[ChunkKind, int] = {
    "system": 0,
    "policy": 1,
    "memory": 2,
    "retrieved": 3,
    "tool_result": 4,
    "history": 5,
}


@runtime_checkable
class CandidateSource(Protocol):
    """Contributes candidate chunks for a task.

    Memory, retrieval and tool results all satisfy this, which is what
    lets the pipeline gather from them without knowing what they are.
    """

    async def candidates(self, task: Task) -> list[Chunk]: ...


@runtime_checkable
class RetractableSource(Protocol):
    """A source that can strike lines from what it serves, so a later turn never sees them.

    Cutting a false line from one window leaves it where it came from: the next turn
    gathers it again. A source that holds its own text (the conversation) takes it out.
    """

    def retract(self, task: Task, chunk: Chunk, lines: list[str]) -> list[str]: ...


@runtime_checkable
class WindowFitter(Protocol):
    """The strategies, as the fit stage runs them (`StrategySelector`)."""

    async def fit(
        self, context: AssembledContext, task: Task | None = None
    ) -> AssembledContext: ...


@runtime_checkable
class ContextAssembler(Protocol):
    """Builds the window for a task within a token budget."""

    async def assemble(
        self, task: Task, budget_tokens: int | None = None, strategies: WindowFitter | None = None
    ) -> AssembledContext: ...

    def retract(self, task: Task, chunk: Chunk, lines: list[str]) -> list[str]:
        """Strike `lines` of `chunk` from whichever source served it; the lines struck."""
        ...


@runtime_checkable
class RelevanceRanker(Protocol):
    """Scores candidate texts against a query (the NOLLM cascade's `rank`)."""

    async def rank(self, text: str, candidates: list[str]) -> list[float]: ...


@instrument("context")
class FiveStagePipeline:
    """Assembly as five small, independently testable stages."""

    def __init__(
        self,
        sources: list[CandidateSource] | None = None,
        ranker: RelevanceRanker | None = None,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        settings = get_settings().context
        self._sources = sources or []
        self._ranker = ranker
        # Two-pass scoring. The ranker scores everything cheaply; the
        # reranker rescores only what survived, with a model that reads
        # query and passage together.
        self._reranker = reranker
        self._rerank = settings.rerank
        self._budget = settings.token_budget
        self._overlap = settings.consolidation.near_duplicate_overlap
        self._logger = get_logger("context")

    @logged("context")
    async def assemble(
        self, task: Task, budget_tokens: int | None = None, strategies: WindowFitter | None = None
    ) -> AssembledContext:
        """Run all five stages and return the assembled window."""
        budget = budget_tokens or self._budget
        candidates = await self.gather_candidates(task)
        scored = await self.score_relevance(task, candidates)
        managed = await self._managed(scored, budget, task, strategies)
        fitted, dropped = self.fit_to_budget(managed.chunks, budget)
        ordered = self.order(self.format(fitted))
        return AssembledContext(
            chunks=ordered,
            budget_tokens=budget,
            used_tokens=sum(chunk.token_estimate for chunk in ordered),
            dropped=[*managed.dropped, *dropped],
            applied_strategies=managed.applied_strategies,
        )

    async def _managed(
        self, chunks: list[Chunk], budget: int, task: Task, strategies: WindowFitter | None
    ) -> AssembledContext:
        """The candidates after the strategies, before anything is cut for the budget."""
        window = AssembledContext(
            chunks=chunks,
            budget_tokens=budget,
            used_tokens=sum(chunk.token_estimate for chunk in chunks),
        )
        return await strategies.fit(window, task) if strategies is not None else window

    def retract(self, task: Task, chunk: Chunk, lines: list[str]) -> list[str]:
        """Ask every source that can strike lines to strike these; the lines that went."""
        return [
            line
            for source in self._sources
            if isinstance(source, RetractableSource)
            for line in source.retract(task, chunk, lines)
        ]

    async def gather_candidates(self, task: Task) -> list[Chunk]:
        """Stage 1 — collect from every registered source, cost-annotated."""
        gathered: list[Chunk] = []
        for source in self._sources:
            gathered.extend(await source.candidates(task))
        return [with_trust(self._with_cost(chunk)) for chunk in gathered]

    async def score_relevance(self, task: Task, chunks: list[Chunk]) -> list[Chunk]:
        """Stage 2 — score each candidate against the task.

        Pinned chunks are scored too but never compete: a system prompt
        that scores poorly against the query is still not optional.
        """
        if not chunks:
            return []
        scores = await self._rank(task.query, [chunk.content for chunk in chunks])
        scored = [
            chunk.model_copy(update={"score": 1.0 if chunk.pinned else score})
            for chunk, score in zip(chunks, scores)
        ]
        return await self._rerank_shortlist(task.query, scored)

    async def _rerank_shortlist(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        """Rescore the best few with the stronger, slower model.

        A shortlist, never the whole set: a cross-encoder runs attention
        over every (query, passage) pair, so its cost is linear in
        candidates and its latency is what stops it being used on a
        corpus. Ten good candidates is where it earns its keep.

        Pinned chunks are excluded. They do not compete on relevance, so
        paying a model to score them would buy a number nothing reads.
        """
        if self._reranker is None:
            return chunks
        pinned = [chunk for chunk in chunks if chunk.pinned]
        competing = sorted(
            (chunk for chunk in chunks if not chunk.pinned),
            key=lambda chunk: chunk.score,
            reverse=True,
        )
        shortlist, remainder = (
            competing[: self._rerank.shortlist],
            competing[self._rerank.shortlist :],
        )
        if len(shortlist) < self._rerank.min_candidates:
            return chunks
        reranked = await self._reranker.rerank(query, shortlist, len(shortlist))
        self._logger.debug(
            "Reranked the shortlist",
            candidates=len(shortlist),
            held_back=len(remainder),
        )
        return [*pinned, *reranked, *remainder]

    def fit_to_budget(self, chunks: list[Chunk], budget: int) -> tuple[list[Chunk], list[str]]:
        """Stage 3 — take the best chunks that fit, reporting what was cut.

        Near-duplicates are dropped before anything else. Two sources
        describing the same order — working memory and the retrieved
        corpus, say — spend the budget twice on one fact and leave the
        model choosing between near-identical options, which is the
        "confusion" failure mode by construction. Highest-scoring copy
        wins, pinned chunks always survive.
        """
        ordered = sorted(chunks, key=lambda chunk: (not chunk.pinned, -chunk.score))
        kept: list[Chunk] = []
        dropped: list[str] = []
        used = 0
        for chunk in ordered:
            if not chunk.pinned and any(
                jaccard(chunk.content, existing.content) > self._overlap for existing in kept
            ):
                dropped.append(chunk.source or chunk.kind)
                continue
            if used + chunk.token_estimate <= budget or chunk.pinned:
                kept.append(chunk)
                used += chunk.token_estimate
            else:
                dropped.append(chunk.source or chunk.kind)
        if dropped:
            self._logger.debug("Dropped chunks over budget", count=len(dropped))
        return kept, dropped

    def format(self, chunks: list[Chunk]) -> list[Chunk]:
        """Stage 4 — label each chunk so its provenance survives assembly.

        An unlabeled retrieved passage is indistinguishable from an
        instruction once it is in the window. The label is what keeps the
        instruction/data boundary legible to the model.
        """
        labelled = [f"{provenance_header(chunk)}\n{chunk.content}" for chunk in chunks]
        # Recounted: the label is sent too, so it is paid for too.
        return [
            chunk.model_copy(update={"content": text, "token_estimate": self.estimate_tokens(text)})
            for chunk, text in zip(chunks, labelled)
        ]

    def order(self, chunks: list[Chunk]) -> list[Chunk]:
        """Stage 5 — place chunks by kind; stable chunks by source, the rest by score."""
        return sorted(chunks, key=_position)

    def estimate_tokens(self, text: str) -> int:
        """A chunk's cost, by the shared tokenizer (never zero: an empty chunk still costs)."""
        return max(1, count_tokens(text))

    def _with_cost(self, chunk: Chunk) -> Chunk:
        """Annotate a chunk with its token cost if it has none."""
        if chunk.token_estimate:
            return chunk
        return chunk.model_copy(update={"token_estimate": self.estimate_tokens(chunk.content)})

    async def _rank(self, query: str, texts: list[str]) -> list[float]:
        """Relevance via the injected cascade, or literal overlap without one."""
        if self._ranker is not None:
            return await self._ranker.rank(query, texts)
        needle = set(query.casefold().split())
        return [
            len(needle & set(text.casefold().split())) / len(needle) if needle else 0.0
            for text in texts
        ]


def _position(chunk: Chunk) -> tuple[int, int, str, float]:
    """Kind first; a stable chunk then by source (fixed turn to turn), any other by score."""
    stable = bool(chunk.metadata.get(STABLE))
    return (
        _KIND_ORDER.get(chunk.kind, len(_KIND_ORDER)),
        0 if stable else 1,
        chunk.source if stable else "",
        0.0 if stable else -chunk.score,
    )
