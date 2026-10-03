"""A run cancelled on purpose keeps its progress and resumes from the last completed step.

Three nodes: `first` records that it ran, `slow` waits until released (the
moment an operator presses Cancel), `last` finishes. Cancelled in `slow`,
the run is `cancelled` with `slow` as its next node; resumed, `first` does
not run again — its checkpoint is the work already done.
"""

import asyncio
from pathlib import Path
from typing import Any, TypedDict

import pytest
from langgraph.graph import END, START, StateGraph

from agent_platform.durability import RunRegistry, create_checkpointer
from agent_platform.durability.runner import DurableGraphRunner
from agent_platform.persistence import SqliteConnectionPool, SqlitePlatformStore


class _State(TypedDict, total=False):
    visited: list[str]


class _Graph:
    """A graph whose middle node blocks until the test releases it."""

    def __init__(self, checkpointer: Any) -> None:
        self.ran: list[str] = []
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        graph = StateGraph(_State)
        graph.add_node("first", self._node("first"))
        graph.add_node("slow", self._slow)
        graph.add_node("last", self._node("last"))
        graph.add_edge(START, "first")
        graph.add_edge("first", "slow")
        graph.add_edge("slow", "last")
        graph.add_edge("last", END)
        self.compiled = graph.compile(checkpointer=checkpointer)

    def _node(self, name: str) -> Any:
        async def run(state: _State) -> _State:
            self.ran.append(name)
            return {"visited": [*state.get("visited", []), name]}

        return run

    async def _slow(self, state: _State) -> _State:
        self.ran.append("slow")
        self.entered.set()
        await self.release.wait()
        return {"visited": [*state.get("visited", []), "slow"]}


@pytest.fixture
def parts(tmp_path: Path) -> tuple[_Graph, DurableGraphRunner, RunRegistry]:
    registry = RunRegistry(SqlitePlatformStore(SqliteConnectionPool(":memory:"), agent_id="t"))
    checkpointer = create_checkpointer(path=tmp_path / "checkpoints.db")
    graph = _Graph(checkpointer)
    return graph, DurableGraphRunner(graph.compiled, checkpointer, registry), registry


async def _start_and_cancel(graph: _Graph, runner: DurableGraphRunner) -> Any:
    started = asyncio.ensure_future(
        runner.start("thread-1", {"visited": []}, session_id="s", subject_id="", preview="q")
    )
    await graph.entered.wait()
    assert runner.cancel("thread-1") is True
    return await started


class TestACancelledRunResumesWhereItStopped:
    async def test_cancel_records_the_run_as_resumable_at_the_node_in_flight(
        self, parts: tuple[_Graph, DurableGraphRunner, RunRegistry]
    ) -> None:
        graph, runner, registry = parts
        outcome = await _start_and_cancel(graph, runner)
        assert outcome.status == "cancelled" and outcome.next_nodes == ["slow"]
        record = registry.get("thread-1")
        assert record is not None and record.status == "cancelled"
        assert record.next_node == "slow"

    async def test_resume_reruns_only_what_did_not_finish(
        self, parts: tuple[_Graph, DurableGraphRunner, RunRegistry]
    ) -> None:
        graph, runner, registry = parts
        await _start_and_cancel(graph, runner)
        graph.release.set()
        outcome = await runner.resume("thread-1")
        assert outcome.status == "completed"
        assert outcome.values["visited"] == ["first", "slow", "last"]
        assert graph.ran == ["first", "slow", "slow", "last"], "first ran once; slow re-ran"

    async def test_nothing_running_cannot_be_cancelled(
        self, parts: tuple[_Graph, DurableGraphRunner, RunRegistry]
    ) -> None:
        _, runner, _ = parts
        assert runner.cancel("no-such-thread") is False

    async def test_a_cancellation_nobody_asked_for_still_propagates(
        self, parts: tuple[_Graph, DurableGraphRunner, RunRegistry]
    ) -> None:
        """Process shutdown cancels the caller: that must not be swallowed as a clean stop."""
        graph, runner, registry = parts
        started = asyncio.ensure_future(
            runner.start("thread-2", {"visited": []}, session_id="s", subject_id="", preview="q")
        )
        await graph.entered.wait()
        started.cancel()
        with pytest.raises(asyncio.CancelledError):
            await started
        record = registry.get("thread-2")
        assert record is not None and record.status == "cancelled"
