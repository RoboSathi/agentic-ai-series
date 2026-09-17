"""Semantic memory stub for Phase 0."""

from agentic_commons.logging import log_call
from agentic_commons.memory.semantic.types import Fact


class SemanticMemory:
    """Semantic memory interface stub."""

    @log_call("memory")
    def store(self, fact: Fact) -> None:
        """Phase 0 stub — no real storage yet."""
        pass

    @log_call("memory")
    def query(self, text: str) -> list[Fact]:
        """Phase 0 stub — no real query yet."""
        return []
