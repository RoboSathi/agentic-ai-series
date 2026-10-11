"""Context assembly, strategies, retrieval and failure detection (LLD §7)."""

from agent_platform.context import (
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
from agent_platform.context.failure import RotMonitor
from agent_platform.evals import create_scoring_cascade

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


TASK = Task(query="refund", session_id="s")


class _Store:
    """An offload store in a dict: what the strategy keeps, and what it hands back."""

    def __init__(self) -> None:
        self.kept: dict[str, str] = {}

    def put(self, session_id: str, text: str) -> str:
        handle = f"h{len(self.kept)}"
        self.kept[handle] = text
        return handle

    def get(self, session_id: str, handle: str) -> str | None:
        return self.kept.get(handle)


class TestStrategies:
    def _context(self, chunks: list[Chunk], budget: int = 10) -> AssembledContext:
        return AssembledContext(
            chunks=chunks,
            budget_tokens=budget,
            used_tokens=sum(chunk.token_estimate for chunk in chunks),
        )

    async def test_consolidate_removes_duplicates(self) -> None:
        context = self._context(
            [Chunk(content="same", token_estimate=5), Chunk(content="same", token_estimate=5)]
        )
        result = await ConsolidateStrategy().apply(context, TASK)
        assert len(result.chunks) == 1
        assert result.applied_strategies and "consolidate" in result.applied_strategies[0]

    async def test_consolidate_keeps_the_more_trusted_copy(self) -> None:
        context = self._context(
            [
                Chunk(content="order 1 shipped", kind="history", token_estimate=3),
                Chunk(content="order 1 shipped", kind="tool_result", token_estimate=3),
            ]
        )
        result = await ConsolidateStrategy().apply(context, TASK)
        assert [chunk.kind for chunk in result.chunks] == ["tool_result"]

    async def test_reduce_shortens_and_says_so(self) -> None:
        """Silent truncation makes a model answer as if it saw everything."""
        context = self._context([Chunk(content="x " * 50, kind="memory", token_estimate=25)])
        result = await ReduceStrategy().apply(context, TASK)
        assert "[reduced: kept" in result.chunks[0].content

    async def test_reduce_keeps_a_transcripts_latest_lines(self) -> None:
        lines = "\n".join(f"user: turn {i}" for i in range(20))
        context = self._context([Chunk(content=lines, kind="history", token_estimate=60)])
        kept = (await ReduceStrategy().apply(context, TASK)).chunks[0].content
        assert "turn 19" in kept and "turn 0\n" not in kept

    async def test_offload_keeps_the_text_under_a_handle_it_names(self) -> None:
        store = _Store()
        context = self._context([Chunk(content="x" * 4000, token_estimate=1000)])
        result = await OffloadStrategy(store, min_tokens=100).apply(context, TASK)
        assert "h0" in result.chunks[0].content and store.kept["h0"] == "x" * 4000
        assert result.used_tokens < 1000

    async def test_offload_without_a_store_offloads_nothing(self) -> None:
        context = self._context([Chunk(content="x" * 4000, token_estimate=1000)])
        result = await OffloadStrategy(None, min_tokens=100).apply(context, TASK)
        assert result.chunks[0].content == "x" * 4000 and result.applied_strategies == []

    async def test_retrieve_keeps_only_the_top_k(self) -> None:
        chunks = [Chunk(content=str(i), score=i / 10, token_estimate=1) for i in range(10)]
        result = await RetrieveStrategy(top_k=3).apply(self._context(chunks), TASK)
        assert len(result.chunks) == 3

    async def test_isolate_keeps_only_the_tasks_kinds(self) -> None:
        context = self._context(
            [Chunk(content="a", kind="policy"), Chunk(content="b", kind="history")]
        )
        task = Task(query="q", kind="repair")
        result = await IsolateStrategy({"repair": ["policy"]}).apply(context, task)
        assert [chunk.kind for chunk in result.chunks] == ["policy"]

    def test_isolate_partitions_by_kind(self) -> None:
        context = self._context(
            [Chunk(content="a", kind="policy"), Chunk(content="b", kind="history")]
        )
        assert set(IsolateStrategy().partition(context)) == {"policy", "history"}

    async def test_cache_marks_the_stable_chunks(self) -> None:
        """Reordering between turns silently destroys the cache saving."""
        chunks = [
            Chunk(content="h", kind="history", score=0.9),
            Chunk(content="p", kind="policy", source="p"),
        ]
        result = await CacheStrategy().apply(self._context(chunks), TASK)
        assert [bool(c.metadata.get("stable")) for c in result.chunks] == [False, True]

    async def test_pinned_chunks_survive_every_strategy(self) -> None:
        pinned = Chunk(content="x" * 4000, token_estimate=1000, pinned=True)
        for strategy in (
            ConsolidateStrategy(),
            ReduceStrategy(),
            OffloadStrategy(_Store(), min_tokens=1),
            RetrieveStrategy(),
        ):
            result = await strategy.apply(self._context([pinned]), TASK)
            assert result.chunks[0].content == pinned.content, type(strategy).__name__

    async def test_the_selector_applies_no_pressure_strategy_once_it_fits(self) -> None:
        context = self._context([Chunk(content="a", token_estimate=1)], budget=100)
        assert (await StrategySelector().fit(context, TASK)).applied_strategies == []

    async def test_the_selector_reduces_an_oversized_window(self) -> None:
        chunks = [Chunk(content=f"{i} " + "x " * 100, token_estimate=50) for i in range(4)]
        fitted = await StrategySelector().fit(self._context(chunks, budget=60), TASK)
        assert fitted.used_tokens < 200

    async def test_the_pipeline_runs_the_strategies_before_it_cuts(self) -> None:
        chunks = [Chunk(content="same text"), Chunk(content="same text")]
        pipeline = FiveStagePipeline(sources=[_Source(chunks)])
        window = await pipeline.assemble(TASK, strategies=StrategySelector())
        assert len(window.chunks) == 1
        assert any("consolidate" in note for note in window.applied_strategies)


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

    async def test_measures_off_task_distraction(self) -> None:
        context = AssembledContext(
            chunks=[Chunk(content="completely unrelated text"), Chunk(content="refund policy")]
        )
        assert await ContextFailureDetector().distraction_ratio(context, "refund") == 0.5

    def test_near_duplicates_signal_confusion(self) -> None:
        context = AssembledContext(
            chunks=[
                Chunk(content="the refund was approved"),
                Chunk(content="the refund was approved"),
            ]
        )
        assert ContextFailureDetector().near_duplicate_ratio(context) == 1.0

    def test_rot_is_judged_against_the_sessions_own_baseline(self) -> None:
        monitor = RotMonitor()
        for score in (1.0, 1.0, 0.5, 0.5):
            monitor.window_used("s", 2900)
            monitor.record("s", score)
        health = monitor.health("s", used_tokens=2900, budget=3000)
        assert health is not None and health.degraded and health.baseline_score == 1.0

    def test_no_rot_below_the_first_checkpoint(self) -> None:
        monitor = RotMonitor()
        for score in (1.0, 1.0, 0.1, 0.1):
            monitor.record("s", score)
        assert monitor.health("s", used_tokens=10, budget=3000) is None


class TestATranscriptIsNotASetOfClaims:
    """Found in a live run: every session reported `health = [clash]`.

    The detector split every line on the first `:`, so a history chunk
    whose lines read `user: …` / `assistant: …` looked like repeated,
    disagreeing claims about a key called "user". A warning that fires on
    every conversation of more than one turn is a warning nobody can act
    on, and it hid the contradictions the check exists to find.
    """

    def test_a_multi_turn_transcript_does_not_contradict_itself(self) -> None:
        context = AssembledContext(
            chunks=[
                Chunk(
                    content=(
                        "user: where is order 10042\n"
                        "assistant: it is on its way\n"
                        "user: what about order 10043\n"
                        "assistant: that one was delivered"
                    ),
                    kind="history",
                    source="working_memory",
                )
            ]
        )
        assert ContextFailureDetector().contradictions(context) == []

    def test_recalled_memory_restates_rather_than_asserts(self) -> None:
        context = AssembledContext(
            chunks=[
                Chunk(content="current_intent: ORDER_STATUS", kind="memory", source="wm"),
                Chunk(content="current_intent: REQUEST_REFUND", kind="memory", source="wm"),
            ]
        )
        assert ContextFailureDetector().contradictions(context) == []

    def test_a_stale_document_disagreeing_with_a_tool_result_still_clashes(self) -> None:
        """The case the check was written for must keep firing."""
        context = AssembledContext(
            chunks=[
                Chunk(content="refund_status: denied", kind="retrieved", source="10042"),
                Chunk(content="refund_status: completed", kind="tool_result", source="tool"),
            ]
        )
        assert ContextFailureDetector().contradictions(context)

    def test_a_transcript_cannot_mask_a_real_clash(self) -> None:
        """Ignoring the transcript must not mean ignoring what is beside it."""
        context = AssembledContext(
            chunks=[
                Chunk(content="user: hello\nassistant: hi", kind="history", source="wm"),
                Chunk(content="status: shipped", kind="retrieved", source="a"),
                Chunk(content="status: cancelled", kind="retrieved", source="b"),
            ]
        )
        assert ContextFailureDetector().contradictions(context)
