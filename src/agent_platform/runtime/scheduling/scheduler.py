"""A bound on how many of a graph's nodes run at once.

The fan-out itself is LangGraph's: an agent sends every ready node as a `Send`, and
LangGraph runs a superstep of them together. What it does not provide is a cap shared across
those sends, so each holds one of `concurrency` permits (`slot()`) for as long as it works.
`concurrency=1` is not a special case: the same sends run one at a time, which is what makes
a sequential run a faithful reproduction of a parallel one.
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


class AsyncDagScheduler:
    """`concurrency` permits, held by each node of a task graph while it runs."""

    def __init__(self, concurrency: int) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        self._concurrency = concurrency
        self._semaphore = asyncio.Semaphore(concurrency)

    @property
    def concurrency(self) -> int:
        return self._concurrency

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """Hold one of the `concurrency` permits for the duration of a block."""
        async with self._semaphore:
            yield
