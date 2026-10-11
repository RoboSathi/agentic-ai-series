"""The research agent's memory customisations (ADR ra-0023).

The platform owns memory; this package holds what is research's own: the
dream event a finished run writes (`run_events`), how a dream reads it
(`consolidators`), and the profile that puts those beside the defaults.
"""

from research_agent.memory.profile import research_memory_profile
from research_agent.memory.run_events import RunEventWriter, RunOutcome

__all__ = ["RunEventWriter", "RunOutcome", "research_memory_profile"]
