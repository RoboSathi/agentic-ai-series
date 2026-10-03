"""Map-reduce — one operation across many items, then combine (E04).

The shape for a corpus that will not fit in one context window. Each item
is handled alone and the partial results are reduced.

The constraint that decides whether this works: an item must be
interpretable *on its own*. If understanding item seven requires having
read item three, mapping over them independently produces confident
nonsense, and you need retrieval or a larger window instead.

Concurrency is bounded by a bulkhead — mapping a thousand items is a
thousand calls, and issuing them all at once is how you take down the
thing you are calling.
"""

import asyncio
from typing import Awaitable, Callable

from agent_platform.logging import get_logger
from agent_platform.resilience import ResourcePoolBulkhead
from design_patterns.types import PatternFamily, PatternRun, Step

Mapper = Callable[[str], Awaitable[str]]
Reducer = Callable[[list[str]], str]


class MapReducePattern:
    """Maps over items with bounded concurrency, then reduces."""

    name: str = "map_reduce"
    family: PatternFamily = "WF"

    def __init__(
        self,
        mapper: Mapper,
        reducer: Reducer,
        bulkhead: ResourcePoolBulkhead | None = None,
    ) -> None:
        self._mapper = mapper
        self._reducer = reducer
        self._bulkhead = bulkhead or ResourcePoolBulkhead()
        self._logger = get_logger("design_patterns")

    async def run(self, task: str, items: list[str] | None = None) -> PatternRun:
        """Map each item, then reduce the partial results."""
        corpus = items if items is not None else task.splitlines()
        mapped = await asyncio.gather(*(self._map_one(item) for item in corpus))
        steps = [
            Step(kind="map", detail=item[:40], output=output)
            for item, output in zip(corpus, mapped)
        ]
        reduced = self._reducer(list(mapped))
        self._logger.debug("Reduced partial results", items=len(corpus))
        return PatternRun(
            pattern=self.name,
            family=self.family,
            output=reduced,
            steps=[*steps, Step(kind="reduce", detail="reduce", output=reduced)],
            metadata={"items": len(corpus)},
        )

    async def _map_one(self, item: str) -> str:
        """Map one item, holding a pool permit while it runs."""
        async with self._bulkhead.acquire("map_reduce"):
            return await self._mapper(item)
