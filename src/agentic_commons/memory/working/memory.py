"""Working memory — what this conversation currently holds (E13/E14, LLD §8).

Working memory *is* the context window: the facts established in this
session that the next turn should not have to re-derive. It is a
short-term store, not a long-term one — nothing here survives the
session, and anything that should survive it goes through the write
pipeline into semantic or episodic memory instead.

Everything is scoped by session. Two conversations running in the same
process — two TUI windows, a test suite, a batch of replayed
transcripts — must not see each other's facts, and a store that ignored
the session id would leak one customer's order into another customer's
window. That is a correctness boundary, not a tidiness one.

Eviction is deliberately naive: oldest first, capped by count, per
session. The real budget decision belongs to `context`, which knows the
token accounting; duplicating it here would put two components in charge
of the same limit.
"""

from typing import Any

from agentic_commons.config import get_settings
from agentic_commons.logging import get_logger, instrument
from agentic_commons.memory.types import (
    MemoryFact,
    MemoryQuery,
    Provenance,
    WriteResult,
)

# Where the session id is carried on a fact. Metadata rather than a field
# on `MemoryFact`: the other three stores are session-agnostic by design,
# and only the active window needs to know which conversation it belongs
# to.
SESSION_KEY = "session_id"

# The session a fact belongs to when nobody said. One shared window is
# the right behaviour for a single-conversation process (a demo script,
# an episode app); a real console always passes its own id.
DEFAULT_SESSION = "default"


@instrument("memory")
class WorkingMemory:
    """The active window for each live session."""

    def __init__(self, capacity: int | None = None) -> None:
        settings = get_settings()
        self._capacity = capacity or settings.context.top_k
        self._facts: dict[str, list[MemoryFact]] = {}
        self._scratch: dict[str, dict[str, Any]] = {}
        self._logger = get_logger("memory")

    def read(self, query: MemoryQuery) -> list[MemoryFact]:
        """Facts active in this session, newest first.

        Newest first because the window is a recency structure: the order
        just discussed matters more than the one discussed four turns
        ago, and a caller taking the first `limit` should get the recent
        ones rather than whatever arrived earliest.
        """
        session = query.session_id or DEFAULT_SESSION
        candidates = [
            fact
            for fact in reversed(self._facts.get(session, []))
            if not query.category or fact.category == query.category
        ]
        if query.text:
            needle = query.text.casefold()
            matched = [fact for fact in candidates if needle in fact.value.casefold()]
            candidates = matched or candidates
        self._logger.debug(
            "Read working memory",
            session_id=session,
            held=len(self._facts.get(session, [])),
            returned=min(len(candidates), query.limit),
            category=query.category,
        )
        return candidates[: query.limit]

    def write(self, fact: MemoryFact, provenance: Provenance) -> WriteResult:
        """Place a fact in this session's window, evicting the oldest if full."""
        session = str(fact.metadata.get(SESSION_KEY) or DEFAULT_SESSION)
        placed = fact.model_copy(
            update={
                "memory_type": "working",
                "provenance": provenance,
                "metadata": {**fact.metadata, SESSION_KEY: session},
            }
        )
        held = self._facts.setdefault(session, [])
        held.append(placed)
        self._logger.debug(
            "Wrote to working memory",
            session_id=session,
            key=placed.key,
            category=placed.category,
            value=placed.value,
            held=len(held),
        )
        while len(held) > self._capacity:
            evicted = held.pop(0)
            self._logger.debug("Evicted from working memory", session_id=session, key=evicted.key)
        return WriteResult(
            route="auto_commit", written=True, fact=placed, reason="in active window"
        )

    def remember(self, key: str, value: Any, session_id: str = DEFAULT_SESSION) -> None:
        """Scratch state for the current turn — cleared with the window."""
        self._scratch.setdefault(session_id or DEFAULT_SESSION, {})[key] = value

    def recall(self, key: str, default: Any = None, session_id: str = DEFAULT_SESSION) -> Any:
        """Read scratch state for the current turn."""
        return self._scratch.get(session_id or DEFAULT_SESSION, {}).get(key, default)

    def clear(self, session_id: str | None = None) -> None:
        """End one session's window, or every session's.

        Called when a conversation ends. Clearing everything is the right
        default for a process that ran one conversation; a console passes
        its own id so that ending one window leaves the others alone.
        """
        if session_id is None:
            self._facts.clear()
            self._scratch.clear()
            return
        self._facts.pop(session_id, None)
        self._scratch.pop(session_id, None)

    def size(self, session_id: str | None = None) -> int:
        """How many facts are active — in one session, or across all."""
        if session_id is None:
            return sum(len(facts) for facts in self._facts.values())
        return len(self._facts.get(session_id, []))

    @property
    def sessions(self) -> list[str]:
        """Every session with an active window."""
        return sorted(self._facts)
