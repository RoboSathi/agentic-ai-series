"""Router — classify, then dispatch to a fixed handler (E04).

A branch whose condition is a classification rather than a predicate. The
categories and their handlers are fixed; only the routing decision is
made at run time.

Classification is deterministic first, reusing the NOLLM cascade's
discipline: exact keyword before anything cleverer. Routing is the
decision most worth keeping cheap, because it happens on every request
and the answer is usually obvious.

Reach for it when the categories are stable and enumerable. When new
categories appear faster than the table is updated, the router silently
sends them to a default that was never designed for them, and that is the
signal the shape has stopped fitting.
"""

from typing import Awaitable, Callable, Protocol, runtime_checkable

from agent_platform.logging import get_logger
from design_patterns.types import PatternFamily, PatternRun, Step

Handler = Callable[[str], Awaitable[str]]


@runtime_checkable
class Classifier(Protocol):
    """Assigns a request to one of the router's known categories."""

    async def classify(self, task: str, categories: list[str]) -> str: ...


class KeywordClassifier:
    """Deterministic routing on configured keywords per category."""

    def __init__(self, keywords: dict[str, list[str]]) -> None:
        self._keywords = {
            category: [word.casefold() for word in words] for category, words in keywords.items()
        }

    async def classify(self, task: str, categories: list[str]) -> str:
        """First category with a keyword in the task; empty if none match."""
        lowered = task.casefold()
        for category in categories:
            if any(word in lowered for word in self._keywords.get(category, [])):
                return category
        return ""


class RouterPattern:
    """Routes a request to one of a fixed set of handlers."""

    name: str = "router"
    family: PatternFamily = "WF"

    def __init__(
        self,
        routes: dict[str, Handler],
        classifier: Classifier,
        fallback: Handler | None = None,
    ) -> None:
        self._routes = routes
        self._classifier = classifier
        self._fallback = fallback
        self._logger = get_logger("design_patterns")

    async def run(self, task: str) -> PatternRun:
        """Classify, then dispatch."""
        category = await self._classifier.classify(task, list(self._routes))
        handler = self._routes.get(category)
        if handler is None:
            if self._fallback is None:
                return PatternRun(
                    pattern=self.name,
                    family=self.family,
                    succeeded=False,
                    output="",
                    steps=[Step(kind="route", detail="unroutable")],
                    metadata={"category": ""},
                )
            category, handler = "fallback", self._fallback
        self._logger.debug("Routed request", category=category)
        output = await handler(task)
        return PatternRun(
            pattern=self.name,
            family=self.family,
            output=output,
            steps=[Step(kind="route", detail=category, output=output)],
            metadata={"category": category},
        )
