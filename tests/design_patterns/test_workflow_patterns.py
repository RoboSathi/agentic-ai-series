"""Workflow patterns: the path is fixed at design time (LLD §22).

Each test asserts the property that *defines* its pattern, not merely
that it runs. A sequence that skipped a stage would still return a
string; what makes it a sequence is that every stage ran, in order.
"""

import asyncio

import pytest

from agentic_commons.context import Chunk, LexicalRetriever
from design_patterns import (
    BranchPattern,
    FanOutFanInPattern,
    FixedLoopPattern,
    HumanInTheLoopPattern,
    MapReducePattern,
    RagPipelinePattern,
    RouterPattern,
    SequencePattern,
)
from design_patterns.workflow.router import KeywordClassifier


def _stage(suffix: str):  # type: ignore[no-untyped-def]
    async def run(value: str) -> str:
        return f"{value}{suffix}"

    return run


class TestSequence:
    async def test_every_stage_runs_in_order(self) -> None:
        pattern = SequencePattern([("a", _stage("-a")), ("b", _stage("-b")), ("c", _stage("-c"))])
        result = await pattern.run("start")
        assert result.output == "start-a-b-c"
        assert [step.detail for step in result.steps] == ["a", "b", "c"]

    async def test_each_stage_feeds_the_next(self) -> None:
        pattern = SequencePattern([("a", _stage("-a")), ("b", _stage("-b"))])
        result = await pattern.run("x")
        assert result.steps[0].output == "x-a"
        assert result.steps[1].output == "x-a-b"

    async def test_an_empty_sequence_passes_the_task_through(self) -> None:
        assert (await SequencePattern([]).run("unchanged")).output == "unchanged"


class TestBranch:
    @pytest.fixture
    def pattern(self) -> BranchPattern:
        return BranchPattern(
            branches=[
                ("refund", lambda t: "refund" in t, _stage(" -> refund desk")),
                ("delivery", lambda t: "parcel" in t, _stage(" -> delivery desk")),
            ],
            otherwise=("general", _stage(" -> general desk")),
        )

    async def test_the_first_matching_branch_wins(self, pattern: BranchPattern) -> None:
        """Order is part of the contract, not an accident."""
        result = await pattern.run("refund for my parcel")
        assert result.metadata["branch"] == "refund"

    async def test_only_one_branch_runs(self, pattern: BranchPattern) -> None:
        result = await pattern.run("refund please")
        assert len(result.steps) == 1

    async def test_the_else_branch_catches_the_rest(self, pattern: BranchPattern) -> None:
        assert (await pattern.run("hello")).metadata["branch"] == "general"

    async def test_without_an_else_an_unmatched_task_fails(self) -> None:
        """Silently succeeding on a task nothing handled would hide it."""
        pattern = BranchPattern(branches=[("x", lambda t: False, _stage(""))])
        assert (await pattern.run("anything")).succeeded is False


class TestFixedLoop:
    async def test_it_runs_exactly_the_configured_number_of_times(self) -> None:
        async def iterate(value: str, index: int) -> str:
            return f"{value}.{index}"

        result = await FixedLoopPattern(iterate, times=4).run("v")
        assert result.metadata["passes"] == 4
        assert result.output == "v.0.1.2.3"

    async def test_it_can_stop_early_without_losing_its_bound(self) -> None:
        """An early exit does not make the worst case unknown."""

        async def iterate(value: str, index: int) -> str:
            return f"{value}+"

        result = await FixedLoopPattern(iterate, times=10, stop_when=lambda v: len(v) >= 3).run("v")
        assert result.metadata["passes"] == 2 and result.metadata["cap"] == 10


class TestFanOutFanIn:
    async def test_every_branch_contributes_to_the_join(self) -> None:
        pattern = FanOutFanInPattern(
            branches=[("a", _stage("-a")), ("b", _stage("-b")), ("c", _stage("-c"))],
            join=lambda parts: " | ".join(sorted(parts)),
        )
        result = await pattern.run("x")
        assert result.output == "x-a | x-b | x-c"

    async def test_branches_run_concurrently(self) -> None:
        """Sequential execution would take three times as long."""

        async def slow(value: str) -> str:
            await asyncio.sleep(0.05)
            return value

        pattern = FanOutFanInPattern(
            branches=[(str(i), slow) for i in range(3)], join=lambda parts: parts[0]
        )
        started = asyncio.get_running_loop().time()
        await pattern.run("x")
        assert asyncio.get_running_loop().time() - started < 0.12

    async def test_a_failing_branch_fails_the_join(self) -> None:
        """A partial answer presented as a complete one is worse than an error."""

        async def boom(value: str) -> str:
            raise RuntimeError("branch failed")

        pattern = FanOutFanInPattern([("ok", _stage("")), ("bad", boom)], join=lambda p: "")
        with pytest.raises(RuntimeError):
            await pattern.run("x")


class TestMapReduce:
    async def test_it_maps_every_item_then_reduces(self) -> None:
        async def upper(item: str) -> str:
            return item.upper()

        pattern = MapReducePattern(upper, reducer=lambda parts: ",".join(parts))
        result = await pattern.run("", items=["a", "b", "c"])
        assert result.output == "A,B,C"
        assert result.metadata["items"] == 3

    async def test_concurrency_is_bounded(self) -> None:
        """Mapping a thousand items must not issue a thousand calls at once."""
        from agentic_commons.resilience import ResourcePoolBulkhead

        peak = {"now": 0, "max": 0}

        async def track(item: str) -> str:
            peak["now"] += 1
            peak["max"] = max(peak["max"], peak["now"])
            await asyncio.sleep(0.01)
            peak["now"] -= 1
            return item

        pattern = MapReducePattern(
            track, reducer=lambda p: "", bulkhead=ResourcePoolBulkhead(pool_size=2)
        )
        await pattern.run("", items=[str(i) for i in range(10)])
        assert peak["max"] <= 2


class TestRouter:
    @pytest.fixture
    def pattern(self) -> RouterPattern:
        return RouterPattern(
            routes={"billing": _stage(" -> billing"), "delivery": _stage(" -> delivery")},
            classifier=KeywordClassifier(
                {"billing": ["refund", "charge"], "delivery": ["parcel", "tracking"]}
            ),
            fallback=_stage(" -> general"),
        )

    async def test_it_dispatches_to_the_classified_handler(self, pattern: RouterPattern) -> None:
        assert (await pattern.run("where is my parcel")).metadata["category"] == "delivery"

    async def test_an_unknown_category_falls_back(self, pattern: RouterPattern) -> None:
        assert (await pattern.run("hello there")).metadata["category"] == "fallback"

    async def test_without_a_fallback_an_unroutable_task_fails(self) -> None:
        pattern = RouterPattern(routes={"a": _stage("")}, classifier=KeywordClassifier({}))
        assert (await pattern.run("anything")).succeeded is False

    async def test_classification_is_deterministic(self, pattern: RouterPattern) -> None:
        """Routing happens on every request; keeping it cheap matters."""
        first = await pattern.run("refund please")
        second = await pattern.run("refund please")
        assert first.metadata["category"] == second.metadata["category"]


class TestHumanInTheLoop:
    async def test_approval_applies_the_change(self) -> None:
        applied: list[str] = []

        async def apply(proposal: str) -> str:
            applied.append(proposal)
            return "applied"

        pattern = HumanInTheLoopPattern(
            prepare=_stage(" (proposed)"), decide=lambda p: _true(), apply=apply
        )
        result = await pattern.run("refund $80")
        assert result.metadata["approved"] is True and applied

    async def test_rejection_applies_nothing(self) -> None:
        """The pause is before the irreversible step, not after it."""
        applied: list[str] = []

        async def apply(proposal: str) -> str:
            applied.append(proposal)
            return "applied"

        pattern = HumanInTheLoopPattern(prepare=_stage(""), decide=lambda p: _false(), apply=apply)
        result = await pattern.run("refund $80")
        assert result.succeeded is False and applied == []

    def test_it_reports_the_staffing_the_queue_implies(self) -> None:
        """E38: 100/hr at 2 minutes each is 3.33 reviewer-hours per hour."""
        pattern = HumanInTheLoopPattern(
            prepare=_stage(""), decide=lambda p: _true(), apply=_stage("")
        )
        assert pattern.reviewers_required(100, 2) == pytest.approx(3.333, abs=0.01)


class TestRagPipeline:
    @pytest.fixture
    def corpus(self) -> list[Chunk]:
        return [
            Chunk(
                content="Refunds are available for thirty days after delivery.", source="refunds"
            ),
            Chunk(
                content="Standard delivery arrives in three to five working days.",
                source="shipping",
            ),
        ]

    async def test_retrieval_happens_before_generation(self, corpus: list[Chunk]) -> None:
        seen: list[int] = []

        async def generate(task: str, chunks: list[Chunk]) -> str:
            seen.append(len(chunks))
            return "answer"

        pattern = RagPipelinePattern(LexicalRetriever(corpus), generate, top_k=2)
        result = await pattern.run("refunds")
        assert seen and seen[0] > 0
        assert result.metadata["retrieved"] > 0

    async def test_retrieving_nothing_is_reported_not_hidden(self, corpus: list[Chunk]) -> None:
        """Generating anyway produces an ungrounded answer that looks grounded."""

        async def generate(task: str, chunks: list[Chunk]) -> str:
            return "made up"

        pattern = RagPipelinePattern(LexicalRetriever(corpus), generate)
        result = await pattern.run("quantum chromodynamics")
        assert result.succeeded is False and result.output == ""


async def _true() -> bool:
    return True


async def _false() -> bool:
    return False
