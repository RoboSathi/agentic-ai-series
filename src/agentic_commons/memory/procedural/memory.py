"""Procedural memory stub for Phase 0."""

from typing import Optional

from agentic_commons.logging import log_call
from agentic_commons.memory.procedural.types import Skill


class ProceduralMemory:
    """Procedural memory interface stub."""

    @log_call("memory")
    def get_skill(self, name: str) -> Optional[Skill]:
        """Phase 0 stub — no real retrieval yet."""
        return None
