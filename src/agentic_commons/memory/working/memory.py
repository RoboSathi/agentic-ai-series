"""Working memory stub for Phase 0."""

from agentic_commons.logging import log_call
from agentic_commons.memory.working.types import Context


class WorkingMemory:
    """Working memory interface stub."""

    @log_call("memory")
    def get(self, session_id: str) -> Context:
        """Phase 0 stub — no real retrieval yet."""
        return Context(session_id=session_id, data={})

    @log_call("memory")
    def set(self, session_id: str, context: Context) -> None:
        """Phase 0 stub — no real storage yet."""
        pass
