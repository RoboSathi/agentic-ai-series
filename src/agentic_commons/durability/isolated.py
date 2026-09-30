"""Run a graph in isolation: only the inputs you pass, on a thread of its own.

Some checks are only worth anything if the checker cannot see what the
thing it checks saw — an independent reviewer who read the author's
drafts is not independent. That property is easy to lose by accident:
share one state object, one thread id or one checkpointer with the run
being checked, and the checker can read the whole conversation.

So this runner builds a *fresh* compiled graph for every run (from a
factory, so no instance can carry state across runs), gives it a new
thread id and no checkpointer, and refuses any input key the caller did
not declare. What the graph can see is exactly the mapping it was handed.
"""

import uuid
from collections.abc import Callable, Mapping
from typing import Any

from agentic_commons.logging import get_logger

GraphFactory = Callable[[], Any]


class UndeclaredInputError(ValueError):
    """An isolated run was handed an input it did not declare."""


class IsolatedGraphRunner:
    """Runs a freshly built graph on explicitly passed inputs only."""

    def __init__(self, factory: GraphFactory, allowed_inputs: frozenset[str]) -> None:
        self._factory = factory
        self._allowed = allowed_inputs
        self._logger = get_logger("persistence")

    @property
    def allowed_inputs(self) -> frozenset[str]:
        return self._allowed

    async def run(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        """Invoke a new graph instance on a new thread; return its final state."""
        undeclared = set(inputs) - self._allowed
        if undeclared:
            raise UndeclaredInputError(f"undeclared isolated inputs: {sorted(undeclared)}")
        thread_id = f"isolated-{uuid.uuid4().hex}"
        self._logger.debug("Isolated run started", thread=thread_id, inputs=sorted(inputs))
        graph = self._factory()
        result = await graph.ainvoke(dict(inputs), {"configurable": {"thread_id": thread_id}})
        return dict(result) if isinstance(result, Mapping) else {"result": result}
