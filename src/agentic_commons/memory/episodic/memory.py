"""Episodic memory stub for Phase 0."""

from agentic_commons.logging import log_call
from agentic_commons.memory.episodic.types import Episode


class EpisodicMemory:
    """Episodic memory interface stub."""

    @log_call("memory")
    def store(self, episode: Episode) -> None:
        """Phase 0 stub — no real storage yet."""
        pass

    @log_call("memory")
    def recall(self, query: str) -> list[Episode]:
        """Phase 0 stub — no real recall yet."""
        return []
