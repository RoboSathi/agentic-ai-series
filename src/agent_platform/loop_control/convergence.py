"""Is the loop getting anywhere? Progress measured as the new evidence each step adds.

A step whose result is unlike everything the loop has already seen moved it forward; a step
whose result restates what it had (the same rows, the same page, a refusal) did not. `gain` is
one minus the result's highest similarity to the evidence so far. `window` steps in a row
below `min_gain` is a plateau: the loop is told to answer from what it has, before the step
cap would have stopped it with nothing more to show.

An agent's own notion of "done" (every question answered, nothing new found) is a
`ConvergenceSignal` it consults beside this; the governor decides.
"""

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
        if not result.strip():
            return 0.0
        seen = evidence[-self._history :]
        if not seen:
            return 1.0
        row = (await self._similarity.matrix([result], seen))[0]
        return max(0.0, 1.0 - max(row))

    def plateaued(self, gains: list[float]) -> bool:
        """The last `window` steps all added less than `min_gain`."""
        window = self._settings.window
        if not self._settings.enabled or len(gains) < window:
            return False
        return all(gain < self._settings.min_gain for gain in gains[-window:])


__all__ = ["ConvergenceMonitor", "ConvergenceSignal"]
