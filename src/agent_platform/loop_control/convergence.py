"""Is the loop getting anywhere? Progress measured as the new evidence each step adds.

A step whose result is unlike everything the loop has already seen moved it forward; a step
whose result restates what it had (the same rows, the same page, a refusal) did not. `gain` is
one minus the result's highest similarity to the evidence so far. `window` steps in a row
below `min_gain` is a plateau: the loop is told to answer from what it has, before the step
cap would have stopped it with nothing more to show.

An agent's own notion of "done" (every question answered, nothing new found) is a
`ConvergenceSignal` it consults beside this; the governor decides.
"""

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from agent_platform.config.settings import ConvergenceSettings
from agent_platform.context.similarity import TextSimilarity


@runtime_checkable
class ConvergenceSignal(Protocol):
    """An agent's own test of whether its goal is met (None: not yet; else the reason)."""

    def converged(self) -> str | None: ...


class ConvergenceMonitor:
    def __init__(
        self, similarity: TextSimilarity, settings: ConvergenceSettings, history: int
    ) -> None:
        self._similarity = similarity
        self._settings = settings
        self._history = history

    async def gain(self, result: str, evidence: list[str]) -> float:
        """How much new evidence `result` adds to `evidence` (1.0: entirely new)."""
        return (await self.gains([result], evidence))[0]

    async def gains(self, results: list[str], evidence: list[str]) -> list[float]:
        """Each result's gain against the same evidence, in one similarity call.

        One call for the lot: asked a result at a time, the evidence window was embedded
        again for every result.
        """
        seen = evidence[-self._history :]
        texts = [result for result in results if result.strip()]
        rows = iter(await self._similarity.matrix(texts, seen) if seen and texts else [])
        return [self._gain_of(result, seen, rows) for result in results]

    @staticmethod
    def _gain_of(result: str, seen: list[str], rows: Iterator[list[float]]) -> float:
        """No result is no gain; anything is all gain against no evidence."""
        if not result.strip():
            return 0.0
        if not seen:
            return 1.0
        return max(0.0, 1.0 - max(next(rows)))

    def plateaued(self, gains: list[float]) -> bool:
        """The last `window` steps all added less than `min_gain`."""
        window = self._settings.window
        if not self._settings.enabled or len(gains) < window:
            return False
        return all(gain < self._settings.min_gain for gain in gains[-window:])


__all__ = ["ConvergenceMonitor", "ConvergenceSignal"]
