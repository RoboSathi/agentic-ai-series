"""An agent's memory profile: how its dreams turn events into memory (ADR 0012).

The whole of an agent's memory customisation is this one object — three
consolidators, one per long-term store. `MemoryProfile.default()` is what
an agent gets by saying nothing: template session summaries, the
deterministic fact patterns, and skills learned from recurring signals,
all driven by configuration. An agent that wants its own vocabulary
replaces one field (`default().replace(episodic=...)`) and keeps the rest.

Everything else about memory — the files, the inbox, governance, the
dream itself — is the platform's, and the same for every agent.
"""

from dataclasses import dataclass, replace

from agent_platform.config import get_settings
from agent_platform.memory.dream.consolidators import (
    DeterministicFacts,
    EpisodicConsolidator,
    RecurringSignalSkills,
    SemanticConsolidator,
    SkillConsolidator,
)
from agent_platform.memory.dream.summaries import SessionSummarizer


@dataclass(frozen=True)
class MemoryProfile:
    """The three questions a dream asks, answered for one agent."""

    episodic: EpisodicConsolidator
    semantic: SemanticConsolidator
    skills: SkillConsolidator

    @classmethod
    def default(cls) -> "MemoryProfile":
        """What an agent that customises nothing remembers."""
        settings = get_settings()
        dream = settings.memory.dream
        return cls(
            episodic=SessionSummarizer(
                entailment=None,
                threshold=settings.hallucination_check.nli_entailment_threshold,
                outcome_chars=dream.outcome_chars,
                lookup_tools=dream.lookup_tools,
            ),
            semantic=DeterministicFacts(),
            skills=RecurringSignalSkills(dream, settings.skills),
        )

    def replace(
        self,
        *,
        episodic: EpisodicConsolidator | None = None,
        semantic: SemanticConsolidator | None = None,
        skills: SkillConsolidator | None = None,
    ) -> "MemoryProfile":
        """This profile with some consolidators swapped for an agent's own."""
        return replace(
            self,
            episodic=episodic or self.episodic,
            semantic=semantic or self.semantic,
            skills=skills or self.skills,
        )
