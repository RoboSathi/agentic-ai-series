"""Context assembly, strategies, retrieval and failure detection (LLD §7)."""

from agentic_commons.context import (
    AssembledContext,
    CacheStrategy,
    Chunk,
    ConsolidateStrategy,
    ContextFailureDetector,
    CrossEncoderReranker,
    DenseRetriever,
    FiveStagePipeline,
    GraphRetriever,
    HybridRetriever,
    IsolateStrategy,
    LexicalRetriever,
    OffloadStrategy,
    ReduceStrategy,
    RetrieveStrategy,
    StrategySelector,
    Task,
    is_identifier_query,
)
from agentic_commons.evals import create_scoring_cascade

_CORPUS = [
    Chunk(content="Order ORD-10042 shipped on March 3 by express courier.", source="orders"),
    Chunk(content="Customers may request a refund within 30 days of delivery.", source="policy"),
    Chunk(
        content="The parcel arrived damaged and the buyer wants their money back.", source="ticket"
    ),
]


class _Source:
    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks

    async def candidates(self, task: Task) -> list[Chunk]:
        return list(self._chunks)


class TestFiveStagePipeline:
    async def test_assembles_within_the_budget(self) -> None:
        pipeline = FiveStagePipeline([_Source(_CORPUS)])
        assembled = await pipeline.assemble(Task(query="refund"), budget_tokens=40)
        assert assembled.used_tokens <= 40

    async def test_reports_what_it_dropped(self) -> None:
        """What the model was *not* shown is the first question to ask."""
        pipeline = FiveStagePipeline([_Source(_CORPUS)])
        assembled = await pipeline.assemble(Task(query="refund"), budget_tokens=15)
        assert assembled.dropped

    async def test_pinned_chunks_survive_the_budget(self) -> None:
        """An agent that drops its own policy is operating without one."""
        pinned = Chunk(content="x" * 4000, kind="policy", source="policy", pinned=True)
        pipeline = FiveStagePipeline([_Source([pinned, *_CORPUS])])
        assembled = await pipeline.assemble(Task(query="refund"), budget_tokens=10)
        assert any(chunk.pinned for chunk in assembled.chunks)

    async def test_ordering_puts_instructions_before_history(self) -> None:
        chunks = [
            Chunk(content="history", kind="history"),
            Chunk(content="system", kind="system"),
        ]
        pipeline = FiveStagePipeline([_Source(chunks)])
        assembled = await pipeline.assemble(Task(query="q"))
        assert assembled.chunks[0].kind == "system"

    async def test_formatting_labels_provenance(self) -> None:
        """An unlabeled passage is indistinguishable from an instruction."""
        pipeline = FiveStagePipeline([_Source([Chunk(content="x", kind="retrieved", source="db")])])
        assembled = await pipeline.assemble(Task(query="q"))
        assert "[retrieved: db]" in assembled.chunks[0].content

    async def test_empty_sources_produce_an_empty_context(self) -> None:
        assembled = await FiveStagePipeline([]).assemble(Task(query="q"))
        assert assembled.chunks == []

    def test_token_estimate_uses_the_configured_ratio(self) -> None:
        assert FiveStagePipeline([]).estimate_tokens("a" * 40) == 10


class TestStrategies:
    def _context(self, chunks: list[Chunk], budget: int = 10) -> AssembledContext:
        return AssembledContext(
            chunks=chunks,
            budget_tokens=budget,
            used_tokens=sum(chunk.token_estimate for chunk in chunks),
        )

    def test_consolidate_removes_duplicates(self) -> None:
        context = self._context(
            [Chunk(content="same", token_estimate=5), Chunk(content="same", token_estimate=5)]
        )
        assert len(ConsolidateStrategy().apply(context).chunks) == 1

    def test_reduce_shortens_and_says_so(self) -> None:
        """Silent truncation makes a model answer as if it saw everything."""
        context = self._context([Chunk(content="x" * 100, token_estimate=25)])
        assert "[reduced]" in ReduceStrategy(keep_ratio=0.5).apply(context).chunks[0].content

    def test_offload_replaces_bulk_with_a_pointer(self) -> None:
        context = self._context([Chunk(content="x" * 4000, token_estimate=1000)])
        result = OffloadStrategy(min_tokens=100).apply(context)
        assert "offloaded" in result.chunks[0].content and result.used_tokens == 1

    def test_retrieve_keeps_only_the_top_k(self) -> None:
        chunks = [Chunk(content=str(i), score=i / 10, token_estimate=1) for i in range(10)]
        assert len(RetrieveStrategy(top_k=3).apply(self._context(chunks)).chunks) == 3

    def test_isolate_partitions_by_kind(self) -> None:
        context = self._context(
            [Chunk(content="a", kind="policy"), Chunk(content="b", kind="history")]
        )
        assert set(IsolateStrategy().partition(context)) == {"policy", "history"}

    def test_cache_gives_the_prefix_a_stable_order(self) -> None:
        """Reordering between turns silently destroys the cache saving."""
        chunks = [
            Chunk(content="h", kind="history", score=0.9),
            Chunk(content="p", kind="policy", source="p"),
            Chunk(content="s", kind="system", source="s"),
        ]
        strategy = CacheStrategy()
        first = strategy.apply(self._context(chunks))
        second = strategy.apply(self._context(list(reversed(chunks))))
        assert strategy.prefix_signature(first) == strategy.prefix_signature(second)

    def test_pinned_chunks_survive_every_strategy(self) -> None:
        pinned = Chunk(content="x" * 4000, token_estimate=1000, pinned=True)
        for strategy in (
            ConsolidateStrategy(),
            ReduceStrategy(),
            OffloadStrategy(),
            RetrieveStrategy(),
        ):
            result = strategy.apply(self._context([pinned]))
            assert result.chunks[0].content == pinned.content, type(strategy).__name__

    def test_the_selector_stops_once_it_fits(self) -> None:
        context = self._context([Chunk(content="a", token_estimate=1)], budget=100)
        assert StrategySelector().fit(context).applied_strategies == []

    def test_the_selector_reduces_an_oversized_window(self) -> None:
        chunks = [Chunk(content="x" * 200, token_estimate=50) for _ in range(4)]
        fitted = StrategySelector().fit(self._context(chunks, budget=60))
        assert fitted.used_tokens < 200


class TestRetrieval:
    async def test_bm25_finds_an_identifier(self) -> None:
        """An embedding of an order number means nothing."""
        found = await LexicalRetriever(_CORPUS).retrieve("ORD-10042", 1)
        assert found[0].source == "orders"

    async def test_bm25_returns_nothing_for_an_absent_term(self) -> None:
        assert await LexicalRetriever(_CORPUS).retrieve("quantum chromodynamics", 3) == []

    def test_identifier_shaped_queries_are_recognised(self) -> None:
        assert is_identifier_query("where is ORD-10042") is True
        assert is_identifier_query("i want my money back") is False

    async def test_dense_retrieval_finds_a_paraphrase(self) -> None:
        dense = DenseRetriever(create_scoring_cascade(), _CORPUS)
        found = await dense.retrieve("I want my money back", 1)
        assert found[0].source in {"ticket", "policy"}

    async def test_hybrid_fuses_both_modes(self) -> None:
        hybrid = HybridRetriever(
            lexical=LexicalRetriever(_CORPUS),
            dense=DenseRetriever(create_scoring_cascade(), _CORPUS),
        )
        assert len(await hybrid.retrieve("refund for a damaged parcel", 2)) == 2

    async def test_graph_retrieval_walks_edges(self) -> None:
        """Some questions are hops, not similarity."""
        graph = GraphRetriever()
        nodes = [Chunk(content=f"node {n}", source=n) for n in ("a", "b", "c")]
        graph.index(nodes, {"a": ["b"], "b": ["c"]})
        assert len(await graph.retrieve("start at a", 3)) == 3

    async def test_reranking_reorders_a_shortlist(self) -> None:
        reranked = await CrossEncoderReranker(create_scoring_cascade()).rerank(
            "money back", list(_CORPUS), 1
        )
        assert len(reranked) == 1

    async def test_an_empty_corpus_returns_nothing(self) -> None:
        assert await LexicalRetriever([]).retrieve("anything", 5) == []


class TestFailureDetection:
    def test_detects_contradictory_claims(self) -> None:
        context = AssembledContext(
            chunks=[
                Chunk(content="status: shipped", source="a"),
                Chunk(content="status: cancelled", source="b"),
            ]
        )
        assert ContextFailureDetector().contradictions(context)

    def test_agreeing_chunks_are_not_a_clash(self) -> None:
        context = AssembledContext(
            chunks=[
                Chunk(content="status: shipped", source="a"),
                Chunk(content="status: shipped", source="b"),
            ]
        )
        assert ContextFailureDetector().contradictions(context) == []

    def test_measures_off_task_distraction(self) -> None:
        context = AssembledContext(
            chunks=[Chunk(content="completely unrelated text"), Chunk(content="refund policy")]
        )
        assert ContextFailureDetector().distraction_ratio(context, {"refund"}) == 0.5

    def test_near_duplicates_signal_confusion(self) -> None:
        context = AssembledContext(
            chunks=[
                Chunk(content="the refund was approved"),
                Chunk(content="the refund was approved"),
            ]
        )
        assert ContextFailureDetector().near_duplicate_ratio(context) == 1.0

    def test_degradation_is_caught_against_the_baseline(self) -> None:
        curve = ContextFailureDetector().degradation_curve({10000: 0.92, 150000: 0.61})
        assert curve[0].degraded is False and curve[-1].degraded is True

    def test_checkpoints_come_from_config(self) -> None:
        assert ContextFailureDetector().checkpoints == [10000, 30000, 60000, 100000, 150000]

    def test_an_empty_curve_is_empty(self) -> None:
        assert ContextFailureDetector().degradation_curve({}) == []
