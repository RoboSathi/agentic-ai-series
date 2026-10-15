"""The six context management strategies (LLD §7, Strategy pattern).

Offload, Reduce, Retrieve, Isolate, Cache, Consolidate. They are not
interchangeable: each trades something different away, and choosing one
means naming what you are willing to lose.

  Offload    — move it out of the window, keep a handle. Loses immediacy.
  Reduce     — shorten it to what the task needs. Loses detail.
  Retrieve   — fetch only what the task needs. Loses what you failed to ask for.
  Isolate    — keep only the kinds this task is about. Loses the rest.
  Cache      — hold the stable part fixed so a provider can cache it. Loses reordering.
  Consolidate— say each thing once. Loses the fact that something was repeated.

They run inside the pipeline's fit stage (`FiveStagePipeline.assemble`), before anything is
cut: the always-on ones on every window, the pressure ones only while it is over budget.
A strategy that changed nothing records nothing — the CONTEXT card lists only what acted.
"""

from abc import ABC, abstractmethod

from agent_platform.config import get_settings
from agent_platform.config.settings import ContextConsolidationSettings
from agent_platform.context.excerpt import relevant_excerpt
from agent_platform.context.offload import OffloadStore, offload_tool_name
from agent_platform.context.similarity import TextSimilarity, jaccard, shared_similarity
from agent_platform.context.trust import chunk_rank
from agent_platform.context.types import OFFLOADED, STABLE, AssembledContext, Chunk, Task
from agent_platform.llm.tokenizer import count_tokens
from agent_platform.logging import get_logger

# The chunk kinds that are the same from turn to turn: what Cache holds fixed.
_STABLE_KINDS = ("system", "policy")


class ContextStrategy(ABC):
    """One way of making a window fit or work better."""

    name: str = "strategy"
    # Runs on every window (True), or only while it is over budget (False).
    always: bool = False

    @abstractmethod
    async def apply(self, context: AssembledContext, task: Task) -> AssembledContext:
        """Return a new context with this strategy applied."""

    def _record(
        self, context: AssembledContext, chunks: list[Chunk], **note: object
    ) -> AssembledContext:
        """Rebuild a context around new chunks, saying what this strategy did."""
        template = get_settings().context.strategy_notes[self.name]
        return context.model_copy(
            update={
                "chunks": chunks,
                "used_tokens": sum(chunk.token_estimate for chunk in chunks),
                "applied_strategies": [*context.applied_strategies, template.format(**note)],
            }
        )


class ConsolidateStrategy(ContextStrategy):
    """Say each thing once — exact copies, near-copies and paraphrases alike.

    Repetition is not free: it consumes budget and, worse, reads to the
    model as emphasis — the mechanism behind context *distraction*. Of two
    copies, the more trusted source's is kept, then the higher-scoring one.
    """

    name = "consolidate"
    always = True

    def __init__(self, similarity: TextSimilarity | None = None) -> None:
        self._similarity = similarity or shared_similarity()

    async def apply(self, context: AssembledContext, task: Task) -> AssembledContext:
        chunks = context.chunks
        if len(chunks) < 2:
            return context
        similar = await self._similarity.matrix(_bodies(chunks), _bodies(chunks))
        dropped = self._duplicates(chunks, similar)
        if not dropped:
            return context
        kept = [chunk for index, chunk in enumerate(chunks) if index not in dropped]
        return self._record(context, kept, count=len(dropped))

    def _duplicates(self, chunks: list[Chunk], similar: list[list[float]]) -> set[int]:
        settings = get_settings().context.consolidation
        order = sorted(range(len(chunks)), key=lambda i: _keep_first(chunks[i]))
        kept: list[int] = []
        dropped: set[int] = set()
        for index in order:
            copy = any(
                _same(chunks[index], chunks[other], similar[index][other], settings)
                for other in kept
            )
            if copy and not chunks[index].pinned:
                dropped.add(index)
            else:
                kept.append(index)
        return dropped


class IsolateStrategy(ContextStrategy):
    """Keep only the kinds this task is about (`context.isolation[task.kind]`).

    The structural answer to context *clash*: two chunks that contradict
    each other cannot confuse a call they were never both in.
    """

    name = "isolate"
    always = True

    def __init__(self, keep_kinds: dict[str, list[str]] | None = None) -> None:
        self._keep_kinds = (
            keep_kinds if keep_kinds is not None else get_settings().context.isolation
        )

    async def apply(self, context: AssembledContext, task: Task) -> AssembledContext:
        allowed = self._keep_kinds.get(task.kind)
        if allowed is None:
            return context
        kept = [chunk for chunk in context.chunks if chunk.pinned or chunk.kind in allowed]
        if len(kept) == len(context.chunks):
            return context
        return self._record(
            context, kept, count=len(context.chunks) - len(kept), kinds=", ".join(allowed)
        )

    def partition(self, context: AssembledContext) -> dict[str, AssembledContext]:
        """Split one context into one sub-context per kind."""
        partitions: dict[str, list[Chunk]] = {}
        for chunk in context.chunks:
            partitions.setdefault(chunk.kind, []).append(chunk)
        return {
            kind: context.model_copy(
                update={
                    "chunks": chunks,
                    "used_tokens": sum(chunk.token_estimate for chunk in chunks),
                }
            )
            for kind, chunks in partitions.items()
        }


class CacheStrategy(ContextStrategy):
    """Hold the stable chunks fixed, first and in a deterministic order.

    A provider's prefix cache needs a byte-identical prefix. Ordering the
    system and policy chunks by score — which changes with every query —
    silently destroys the saving. These are marked stable: the pipeline
    orders them by source, and the request builder counts them into the
    cacheable prefix.
    """

    name = "cache"
    always = True

    async def apply(self, context: AssembledContext, task: Task) -> AssembledContext:
        stable = [
            c for c in context.chunks if c.kind in _STABLE_KINDS and not c.metadata.get(STABLE)
        ]
        if not stable:
            return context
        chunks = [
            chunk.model_copy(update={"metadata": {**chunk.metadata, STABLE: True}})
            if chunk.kind in _STABLE_KINDS
            else chunk
            for chunk in context.chunks
        ]
        return self._record(context, chunks, count=len(stable))


class RetrieveStrategy(ContextStrategy):
    """Keep pinned chunks and only the best-scoring k of the rest."""

    name = "retrieve"

    def __init__(self, top_k: int | None = None) -> None:
        self._top_k = top_k or get_settings().context.top_k

    async def apply(self, context: AssembledContext, task: Task) -> AssembledContext:
        pinned = [chunk for chunk in context.chunks if chunk.pinned]
        rest = sorted(
            (chunk for chunk in context.chunks if not chunk.pinned),
            key=lambda chunk: chunk.score,
            reverse=True,
        )
        if len(rest) <= self._top_k:
            return context
        return self._record(context, pinned + rest[: self._top_k], count=len(rest) - self._top_k)


class OffloadStrategy(ContextStrategy):
    """Move bulky chunks out of the window, leaving a handle the model can fetch them by.

    Without a store there is nowhere to keep the text, so nothing is offloaded:
    a pointer to nothing would be a deletion that claims otherwise.
    """

    name = "offload"

    def __init__(self, store: OffloadStore | None = None, min_tokens: int | None = None) -> None:
        settings = get_settings().context
        self._store = store
        self._min_tokens = min_tokens or int(
            settings.token_budget * settings.offload.min_tokens_fraction
        )

    async def apply(self, context: AssembledContext, task: Task) -> AssembledContext:
        if self._store is None:
            return context
        chunks: list[Chunk] = []
        handles: list[str] = []
        for chunk in context.chunks:
            if chunk.pinned or chunk.token_estimate < self._min_tokens:
                chunks.append(chunk)
                continue
            handle = self._store.put(task.session_id, chunk.content)
            handles.append(handle)
            chunks.append(_placeholder(chunk, handle))
        if not handles:
            return context
        return self._record(context, chunks, count=len(handles), handles=", ".join(handles))


class ReduceStrategy(ContextStrategy):
    """Shorten what is left, by what each kind is: the task's passages, a transcript's tail.

    The cut is stated in the text rather than performed silently — a model that
    cannot tell it received part of a document answers as though it had all of it.
    """

    name = "reduce"

    async def apply(self, context: AssembledContext, task: Task) -> AssembledContext:
        settings = get_settings().context.reduce
        reduced: list[Chunk] = []
        changed = 0
        for chunk in context.chunks:
            mode = settings.by_kind.get(chunk.kind)
            shorter = chunk if chunk.pinned or mode is None else _shorten(chunk, mode, task)
            changed += shorter is not chunk
            reduced.append(shorter)
        if not changed:
            return context
        return self._record(context, reduced, count=changed)


def reduce_history(context: AssembledContext) -> AssembledContext:
    """Rot's mitigation: the transcript cut to its most recent lines, whatever the budget."""
    chunks = [
        _shorten(chunk, "tail", Task(query="")) if chunk.kind == "history" else chunk
        for chunk in context.chunks
    ]
    return context.model_copy(
        update={"chunks": chunks, "used_tokens": sum(chunk.token_estimate for chunk in chunks)}
    )


class StrategySelector:
    """Applies the enabled strategies (Facade).

    The always-on strategies first (consolidate, isolate, cache), then the
    pressure ones in order of least loss — drop the least relevant, offload
    bulk, and only then reduce — each only while the window is over budget.
    """

    def __init__(
        self,
        strategies: list[ContextStrategy] | None = None,
        *,
        offload_store: OffloadStore | None = None,
        similarity: TextSimilarity | None = None,
    ) -> None:
        toggles = get_settings().context.strategies
        available: list[ContextStrategy] = [
            ConsolidateStrategy(similarity),
            IsolateStrategy(),
            CacheStrategy(),
            RetrieveStrategy(),
            OffloadStrategy(offload_store),
            ReduceStrategy(),
        ]
        self._strategies = strategies or [
            strategy for strategy in available if getattr(toggles, strategy.name, False)
        ]
        self._offloads = offload_store is not None and any(
            isinstance(strategy, OffloadStrategy) for strategy in self._strategies
        )
        self._logger = get_logger("context")

    @property
    def offloads(self) -> bool:
        """Whether this selector can offload — and so whether its fetch tool should be offered."""
        return self._offloads

    async def fit(self, context: AssembledContext, task: Task | None = None) -> AssembledContext:
        """Apply the always-on strategies, then the pressure ones while over budget."""
        current, task = context, task or Task(query="")
        for strategy in self._strategies:
            if not strategy.always and current.within_budget:
                continue
            before = len(current.applied_strategies)
            current = await strategy.apply(current, task)
            if len(current.applied_strategies) > before:
                self._logger.debug(
                    "Applied context strategy",
                    strategy=strategy.name,
                    note=current.applied_strategies[-1],
                    used=current.used_tokens,
                    budget=current.budget_tokens,
                )
        return current


def _bodies(chunks: list[Chunk]) -> list[str]:
    return [chunk.content for chunk in chunks]


def _keep_first(chunk: Chunk) -> tuple[bool, int, float]:
    """Which of two copies survives: pinned, then the more trusted, then the better scored."""
    return (not chunk.pinned, -chunk_rank(chunk), -chunk.score)


def _same(
    left: Chunk, right: Chunk, similarity: float, settings: ContextConsolidationSettings
) -> bool:
    """An exact copy, a near-copy by word overlap, or a paraphrase by similarity."""
    exact = left.content.strip().casefold() == right.content.strip().casefold()
    near = jaccard(left.content, right.content) > settings.near_duplicate_overlap
    return exact or near or similarity >= settings.semantic_threshold


def _placeholder(chunk: Chunk, handle: str) -> Chunk:
    settings = get_settings().context.offload
    text = settings.placeholder.format(
        source=chunk.source or chunk.kind,
        tokens=chunk.token_estimate,
        tool=offload_tool_name(),
        handle=handle,
    )
    metadata = {**chunk.metadata, OFFLOADED: handle}
    return chunk.model_copy(
        update={"content": text, "token_estimate": count_tokens(text), "metadata": metadata}
    )


def _shorten(chunk: Chunk, mode: str, task: Task) -> Chunk:
    """The chunk cut to `keep_ratio` of its length, by mode, with the cut stated."""
    settings = get_settings().context.reduce
    limit = max(1, int(len(chunk.content) * settings.keep_ratio))
    if limit >= len(chunk.content):
        return chunk
    if mode == "excerpt":
        kept = relevant_excerpt(chunk.content, task.query, limit, settings.excerpt_window_chars)
    elif mode == "tail":
        kept = _tail(chunk.content, limit)
    else:
        kept = chunk.content[:limit]
    tokens = count_tokens(kept)
    text = f"{kept}\n{settings.marker.format(kept=tokens, total=chunk.token_estimate)}"
    return chunk.model_copy(update={"content": text, "token_estimate": count_tokens(text)})


def _tail(text: str, limit: int) -> str:
    """The most recent whole lines that fit in `limit` characters."""
    kept: list[str] = []
    used = 0
    for line in reversed(text.splitlines()):
        if used + len(line) > limit and kept:
            break
        kept.append(line)
        used += len(line) + 1
    return "\n".join(reversed(kept))


__all__ = [
    "CacheStrategy",
    "ConsolidateStrategy",
    "ContextStrategy",
    "IsolateStrategy",
    "OffloadStrategy",
    "ReduceStrategy",
    "RetrieveStrategy",
    "StrategySelector",
    "reduce_history",
]
