"""The async DAG scheduler: waves, bounded concurrency, failure isolation."""

import asyncio

import pytest

from agent_platform.scheduling import (
    AsyncDagScheduler,
    InvalidTaskGraphError,
    TaskGraph,
    TaskNode,
    WaveSummary,
    find_cycle,
)


def _graph(*edges: tuple[str, tuple[str, ...]]) -> TaskGraph:
    return TaskGraph([TaskNode(node_id=node, depends_on=deps) for node, deps in edges])


class TestTaskGraph:
    def test_waves_follow_dependencies(self) -> None:
        graph = _graph(("a", ()), ("b", ()), ("c", ("a", "b")), ("d", ("c",)))
        assert graph.waves() == [["a", "b"], ["c"], ["d"]]

    def test_a_cycle_is_refused_and_named(self) -> None:
        with pytest.raises(InvalidTaskGraphError) as raised:
            _graph(("a", ("c",)), ("b", ("a",)), ("c", ("b",)))
        assert set(raised.value.cycle) == {"a", "b", "c"}

    def test_find_cycle_is_none_for_a_dag(self) -> None:
        assert find_cycle([TaskNode(node_id="a"), TaskNode(node_id="b", depends_on=("a",))]) is None

    def test_an_undeclared_dependency_is_refused(self) -> None:
        with pytest.raises(InvalidTaskGraphError, match="undeclared"):
            _graph(("a", ("ghost",)))

    def test_duplicate_ids_are_refused(self) -> None:
        with pytest.raises(InvalidTaskGraphError, match="duplicate"):
            TaskGraph([TaskNode(node_id="a"), TaskNode(node_id="a")])

    def test_blocked_lists_dependents_of_a_failure(self) -> None:
        graph = _graph(("a", ()), ("b", ("a",)), ("c", ()))
        assert graph.blocked({"a": "failed"}) == ["b"]
        assert graph.ready({"a": "failed"}) == ["c"]


class TestAsyncDagScheduler:
    async def test_every_node_runs_after_its_parents(self) -> None:
        order: list[str] = []

        async def worker(node_id: str) -> str:
            order.append(node_id)
            return node_id.upper()

        outcomes = await AsyncDagScheduler(2).run(_graph(("a", ()), ("b", ("a",))), worker)
        assert order == ["a", "b"]
        assert outcomes["b"].value == "B" and outcomes["b"].wave == 2

    async def test_concurrency_is_bounded(self) -> None:
        in_flight = 0
        peak = 0

        async def worker(node_id: str) -> None:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1

        graph = TaskGraph([TaskNode(node_id=str(index)) for index in range(6)])
        await AsyncDagScheduler(2).run(graph, worker)
        assert peak == 2

    async def test_concurrency_one_is_sequential_with_the_same_outcomes(self) -> None:
        async def worker(node_id: str) -> str:
            return node_id

        graph = _graph(("a", ()), ("b", ()), ("c", ("a",)))
        parallel = await AsyncDagScheduler(3).run(graph, worker)
        sequential = await AsyncDagScheduler(1).run(graph, worker)
        assert {k: v.status for k, v in parallel.items()} == {
            k: v.status for k, v in sequential.items()
        }

    async def test_a_failure_blocks_dependents_and_nothing_else(self) -> None:
        async def worker(node_id: str) -> None:
            if node_id == "a":
                raise RuntimeError("boom")

        graph = _graph(("a", ()), ("b", ("a",)), ("c", ("b",)), ("d", ()))
        outcomes = await AsyncDagScheduler(2).run(graph, worker)
        assert outcomes["a"].status == "failed" and "boom" in outcomes["a"].error
        assert outcomes["b"].status == "blocked"
        assert outcomes["c"].status == "blocked"
        assert outcomes["d"].status == "done"

    async def test_the_caller_can_stop_between_waves(self) -> None:
        async def worker(node_id: str) -> None:
            return None

        waves: list[WaveSummary] = []
        graph = _graph(("a", ()), ("b", ("a",)))
        outcomes = await AsyncDagScheduler(1, observer=waves.append).run(
            graph, worker, should_continue=lambda: not waves
        )
        assert outcomes["a"].status == "done"
        assert outcomes["b"].status == "skipped"

    def test_concurrency_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            AsyncDagScheduler(0)
