"""The six context management strategies (E18, LLD §7, Strategy pattern).

Offload, Reduce, Retrieve, Isolate, Cache, Consolidate. They are not
interchangeable: each trades something different away, and choosing one
means naming what you are willing to lose.

  Offload    — move it out of the window, keep a pointer. Loses immediacy.
  Reduce     — summarize. Loses detail, permanently.
  Retrieve   — fetch only what the task needs. Loses what you failed to ask for.
  Isolate    — separate concerns into their own windows. Costs a round trip.
  Cache      — reuse a stable prefix. Loses the ability to vary that prefix.
  Consolidate— merge duplicates. Loses the fact that something was repeated.
"""

from abc import ABC, abstractmethod

from agentic_commons.config import get_settings
from agentic_commons.context.types import AssembledContext, Chunk
from agentic_commons.logging import get_logger


class ContextStrategy(ABC):
    """One way of making a window fit or work better."""

    name: str = "strategy"

    @abstractmethod
    def apply(self, context: AssembledContext) -> AssembledContext:
        """Return a new context with this strategy applied."""

    def _record(self, context: AssembledContext, chunks: list[Chunk]) -> AssembledContext:
        """Rebuild a context around new chunks, noting that this ran."""
        return context.model_copy(
            update={
                "chunks": chunks,
                "used_tokens": sum(chunk.token_estimate for chunk in chunks),
                "applied_strategies": [*context.applied_strategies, self.name],
            }
        )


class OffloadStrategy(ContextStrategy):
    """Replace bulky content with a pointer to where it lives.

    The agent can fetch it back if the task turns out to need it, and
    pays nothing for it when it does not.
    """

    name = "offload"

    def __init__(self, min_tokens: int | None = None) -> None:
        self._min_tokens = min_tokens or get_settings().context.token_budget // 4

    def apply(self, context: AssembledContext) -> AssembledContext:
        """Swap oversized unpinned chunks for a reference line."""
        offloaded = [
            chunk
            if chunk.pinned or chunk.token_estimate < self._min_tokens
            else chunk.model_copy(
                update={
                    "content": f"[offloaded: {chunk.source or chunk.kind}, "
                    f"{chunk.token_estimate} tokens — fetch if needed]",
                    "token_estimate": 1,
                }
            )
            for chunk in context.chunks
        ]
        return self._record(context, offloaded)


class ReduceStrategy(ContextStrategy):
    """Keep a leading fraction of each long chunk.

    Truncation is stated honestly in the text rather than performed
    silently — a model that cannot tell it received half a document will
    answer as though it received all of it.
    """

    name = "reduce"

    def __init__(self, keep_ratio: float | None = None) -> None:
        settings = get_settings().context
        self._keep_ratio = keep_ratio if keep_ratio is not None else settings.reduce_keep_ratio

    def apply(self, context: AssembledContext) -> AssembledContext:
        """Shorten unpinned chunks to the configured fraction."""
        reduced = [chunk if chunk.pinned else self._shorten(chunk) for chunk in context.chunks]
        return self._record(context, reduced)

    def _shorten(self, chunk: Chunk) -> Chunk:
        keep = max(1, int(len(chunk.content) * self._keep_ratio))
        if keep >= len(chunk.content):
            return chunk
        return chunk.model_copy(
            update={
                "content": f"{chunk.content[:keep]}… [reduced]",
                "token_estimate": max(1, int(chunk.token_estimate * self._keep_ratio)),
            }
        )


class RetrieveStrategy(ContextStrategy):
    """Keep only the top-scoring chunks, up to the configured k."""

    name = "retrieve"

    def __init__(self, top_k: int | None = None) -> None:
        self._top_k = top_k or get_settings().context.top_k

    def apply(self, context: AssembledContext) -> AssembledContext:
        """Retain pinned chunks plus the best k of the rest."""
        pinned = [chunk for chunk in context.chunks if chunk.pinned]
        rest = sorted(
            (chunk for chunk in context.chunks if not chunk.pinned),
            key=lambda chunk: chunk.score,
            reverse=True,
        )
        return self._record(context, pinned + rest[: self._top_k])


class IsolateStrategy(ContextStrategy):
    """Split the window by concern so unrelated material cannot interfere.

    This is the structural answer to context *clash*: two chunks that
    contradict each other cannot confuse a single call if they were never
    in the same call.
    """

    name = "isolate"

    def __init__(self, keep_kinds: set[str] | None = None) -> None:
        self._keep_kinds = keep_kinds

    def apply(self, context: AssembledContext) -> AssembledContext:
        """Keep only the kinds this call is about."""
        if self._keep_kinds is None:
            return context
        kept = [chunk for chunk in context.chunks if chunk.pinned or chunk.kind in self._keep_kinds]
        return self._record(context, kept)

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
                    "applied_strategies": [*context.applied_strategies, self.name],
                }
            )
            for kind, chunks in partitions.items()
        }


class CacheStrategy(ContextStrategy):
    """Hold the stable prefix constant so a provider cache can hit it.

    Cache hits require a byte-identical prefix. Reordering the system
    and policy chunks between turns — which any score-based ordering will
    do — silently destroys the saving, so this strategy fixes them first
    and in a stable order.
    """

    name = "cache"

    def __init__(self) -> None:
        self._prefix_kinds = ("system", "policy")
        self._logger = get_logger("context")

    def apply(self, context: AssembledContext) -> AssembledContext:
        """Move stable kinds to the front in a deterministic order."""
        prefix = [
            chunk
            for kind in self._prefix_kinds
            for chunk in sorted(
                (c for c in context.chunks if c.kind == kind), key=lambda c: c.source
            )
        ]
        rest = [chunk for chunk in context.chunks if chunk.kind not in self._prefix_kinds]
        return self._record(context, prefix + rest)

    def prefix_signature(self, context: AssembledContext) -> str:
        """The cacheable prefix, for asserting it did not change between turns."""
        return "\n".join(
            chunk.content for chunk in context.chunks if chunk.kind in self._prefix_kinds
        )


class ConsolidateStrategy(ContextStrategy):
    """Merge duplicate content so the window says each thing once.

    Repetition is not free: it consumes budget and, worse, reads to the
    model as emphasis — the mechanism behind context *distraction*.
    """

    name = "consolidate"

    def apply(self, context: AssembledContext) -> AssembledContext:
        """Drop chunks whose content has already appeared."""
        seen: set[str] = set()
        merged: list[Chunk] = []
        for chunk in context.chunks:
            fingerprint = chunk.content.strip().casefold()
            if fingerprint in seen and not chunk.pinned:
                continue
            seen.add(fingerprint)
            merged.append(chunk)
        return self._record(context, merged)


class StrategySelector:
    """Applies the enabled strategies until the context fits (Facade).

    Ordered cheapest-loss first: deduplicate, then drop the least
    relevant, then offload bulk, and only then reduce — because reduction
    is the only one of the four that destroys information outright.
    """

    def __init__(self, strategies: list[ContextStrategy] | None = None) -> None:
        toggles = get_settings().context.strategies
        available: list[ContextStrategy] = [
            ConsolidateStrategy(),
            RetrieveStrategy(),
            OffloadStrategy(),
            ReduceStrategy(),
        ]
        self._strategies = strategies or [
            strategy for strategy in available if getattr(toggles, strategy.name, False)
        ]
        self._logger = get_logger("context")

    def fit(self, context: AssembledContext) -> AssembledContext:
        """Apply strategies in order, stopping as soon as the window fits."""
        current = context
        for strategy in self._strategies:
            if current.within_budget:
                return current
            current = strategy.apply(current)
            self._logger.debug(
                "Applied context strategy",
                strategy=strategy.name,
                used=current.used_tokens,
                budget=current.budget_tokens,
            )
        return current
