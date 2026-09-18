"""Working memory — a pointer, not a store (E13/E14, LLD §8).

Working memory *is* the context window. Giving it its own backing store
would create a second copy of the truth that drifts from what the model
actually sees, which is the bug this class exists to prevent. It holds
references and delegates assembly to `context`.
"""

from typing import Any

from agentic_commons.config import get_settings
from agentic_commons.logging import get_logger
from agentic_commons.memory.types import (
    MemoryFact,
    MemoryQuery,
    Provenance,
    WriteResult,
)


class WorkingMemory:
    """What is active right now: the current turn's facts, in order.

    Eviction is deliberately naive — oldest first, capped by count. The
    real budget decision belongs to `context`, which knows the token
    accounting; duplicating it here would put two components in charge of
    the same limit.
    """

    def __init__(self, capacity: int | None = None) -> None:
        self._enabled = get_settings().memory.working.enabled
        self._capacity = capacity or get_settings().context.top_k
        self._facts: list[MemoryFact] = []
        self._scratch: dict[str, Any] = {}
        self._logger = get_logger("memory")

    def read(self, query: MemoryQuery) -> list[MemoryFact]:
        """Facts currently in the window, newest first, filtered by category."""
        candidates = [
            fact
            for fact in reversed(self._facts)
            if not query.category or fact.category == query.category
        ]
        return candidates[: query.limit]

    def write(self, fact: MemoryFact, provenance: Provenance) -> WriteResult:
        """Place a fact in the active window, evicting the oldest if full."""
        if not self._enabled:
            return WriteResult(route="discard", written=False, reason="working memory disabled")
        placed = fact.model_copy(update={"memory_type": "working", "provenance": provenance})
        self._facts.append(placed)
        if len(self._facts) > self._capacity:
            evicted = self._facts.pop(0)
            self._logger.debug("Evicted from working memory", key=evicted.key)
        return WriteResult(
            route="auto_commit", written=True, fact=placed, reason="in active window"
        )

    def remember(self, key: str, value: Any) -> None:
        """Scratch state for the current turn — cleared with the window."""
        self._scratch[key] = value

    def recall(self, key: str, default: Any = None) -> Any:
        """Read scratch state for the current turn."""
        return self._scratch.get(key, default)

    def clear(self) -> None:
        """Drop everything — the turn, and the window, are over."""
        self._facts.clear()
        self._scratch.clear()

    @property
    def size(self) -> int:
        """How many facts are currently active."""
        return len(self._facts)
