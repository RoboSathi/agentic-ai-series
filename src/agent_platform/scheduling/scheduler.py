"""Run a task graph batch by batch, concurrently, under a bounded semaphore.

Every node whose dependencies are done runs at once, up to `concurrency`
in flight. A node that raises is isolated: it is marked `failed`, its
dependents are marked `blocked`, and every independent branch carries
on. One bad sub-task must cost its own subtree, not the whole run.

`concurrency=1` is not a special case: the same code path runs the same
batches one node at a time, which is what makes a sequential run a
faithful reproduction of a parallel one.

The semaphore is exposed (`slot()`) so a caller that fans work out some
other way — LangGraph `Send`s, for instance — can hold the same bound.
"""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from typing import Any

from agent_platform.logging import get_logger
from agent_platform.scheduling.graph import TaskGraph
from agent_platform.scheduling.types import BatchSummary, OutcomeStatus, TaskOutcome

Worker = Callable[[str], Awaitable[Any]]
ShouldContinue = Callable[[], bool]
BatchObserver = Callable[[BatchSummary], None]


class AsyncDagScheduler:
    """Batch iteration with bounded concurrency and failure isolation."""

    def __init__(self, concurrency: int, observer: BatchObserver | None = None) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        self._concurrency = concurrency
        self._semaphore = asyncio.Semaphore(concurrency)
        self._observer = observer
        self._logger = get_logger("scheduler")

    @property
    def concurrency(self) -> int:
        return self._concurrency

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """Hold one of the `concurrency` permits for the duration of a block."""
        async with self._semaphore:
            yield

    async def run(
        self,
        graph: TaskGraph,
        worker: Worker,
        should_continue: ShouldContinue | None = None,
    ) -> dict[str, TaskOutcome]:
        """Run every node that can run; return how each one ended."""
        outcomes: dict[str, TaskOutcome] = {}
        batch = 0
        while True:
            self._mark_blocked(graph, outcomes, batch)
            ready = graph.ready(_statuses(outcomes))
            if not ready:
                return outcomes
            if should_continue is not None and not should_continue():
                return self._skip_rest(graph, outcomes, batch)
            batch += 1
            outcomes.update(await self.run_batch(ready, worker, batch))

    async def run_batch(
        self, node_ids: list[str], worker: Worker, batch: int
    ) -> dict[str, TaskOutcome]:
        """Run one batch concurrently; every node yields an outcome, none raises."""
        self._logger.info(
            "Batch started", batch=batch, nodes=node_ids, concurrency=self._concurrency
        )
        results = await asyncio.gather(*(self._one(node_id, worker, batch) for node_id in node_ids))
        outcomes = {outcome.node_id: outcome for outcome in results}
        self._observe(batch, outcomes)
        return outcomes

    async def _one(self, node_id: str, worker: Worker, batch: int) -> TaskOutcome:
        async with self.slot():
            try:
                value = await worker(node_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._logger.warning(
                    "Node failed; dependents blocked", node=node_id, error=str(exc)
                )
                return TaskOutcome(node_id=node_id, status="failed", error=str(exc), batch=batch)
        return TaskOutcome(node_id=node_id, status="done", value=value, batch=batch)

    def _mark_blocked(self, graph: TaskGraph, outcomes: dict[str, TaskOutcome], batch: int) -> None:
        # Repeated until stable: a blocked node blocks its own dependents.
        while blocked := graph.blocked(_statuses(outcomes)):
            for node_id in blocked:
                outcomes[node_id] = TaskOutcome(
                    node_id=node_id,
                    status="blocked",
                    error="a dependency did not finish",
                    batch=batch,
                )

    def _skip_rest(
        self, graph: TaskGraph, outcomes: dict[str, TaskOutcome], batch: int
    ) -> dict[str, TaskOutcome]:
        for node_id in graph.node_ids:
            outcomes.setdefault(
                node_id, TaskOutcome(node_id=node_id, status="skipped", batch=batch)
            )
        self._logger.info("Scheduling stopped by caller", batch=batch)
        return outcomes

    def _observe(self, batch: int, outcomes: Mapping[str, TaskOutcome]) -> None:
        summary = BatchSummary(
            batch=batch,
            node_ids=list(outcomes),
            done=sum(outcome.status == "done" for outcome in outcomes.values()),
            failed=sum(outcome.status == "failed" for outcome in outcomes.values()),
        )
        if self._observer is not None:
            self._observer(summary)


def _statuses(outcomes: Mapping[str, TaskOutcome]) -> dict[str, OutcomeStatus]:
    return {node_id: outcome.status for node_id, outcome in outcomes.items()}
