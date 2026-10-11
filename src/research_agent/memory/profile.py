"""The research agent's memory profile: the default, reading runs its own way."""

from agent_platform.memory import MemoryProfile
from agent_platform.memory.dream import CombinedFacts
from research_agent.config import get_agent_settings
from research_agent.memory.consolidators import RunSummary, SourceReliability


def research_memory_profile() -> MemoryProfile:
    """Runs summarised from their record, sources remembered; skills by default."""
    settings = get_agent_settings().memory
    default = MemoryProfile.default()
    return default.replace(
        episodic=RunSummary(settings, fallback=default.episodic),
        semantic=CombinedFacts(default.semantic, SourceReliability(settings)),
    )
