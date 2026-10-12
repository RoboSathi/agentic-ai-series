"""Dreams: consolidating an agent's dream events into long-term memory (ADR 0012).

A completed turn (or run) writes a `DreamEvent` to the agent's inbox,
`data/<package>/memory/dream_events/`. `make dream <agent>` runs a `Dream`, which
asks the agent's `MemoryProfile` what the pending events amount to —
summaries, facts, skills — and writes them through governed memory.
"""

from agent_platform.memory.dream.consolidators import (
    CombinedFacts,
    DeterministicFacts,
    EpisodicConsolidator,
    FactCandidate,
    RecurringSignalSkills,
    SemanticConsolidator,
    SkillCandidate,
    SkillConsolidator,
)
from agent_platform.memory.dream.events import (
    DreamEvent,
    DreamEventSink,
    DreamSignal,
    InboxCounts,
    MemoryAction,
    MemoryObservation,
)
from agent_platform.memory.dream.inbox import FileDreamInbox
from agent_platform.memory.dream.report import (
    DreamEntry,
    DreamReport,
    ReconcileOp,
    change_tree_lines,
)
from agent_platform.memory.dream.service import Dream
from agent_platform.memory.dream.summaries import (
    SUMMARY_CATEGORY,
    DialogueSummarizer,
    SessionSummarizer,
    SessionSummary,
)
from agent_platform.memory.dream.turns import TurnRecorder

__all__ = [
    "SUMMARY_CATEGORY",
    "CombinedFacts",
    "DeterministicFacts",
    "DialogueSummarizer",
    "Dream",
    "DreamEntry",
    "DreamEvent",
    "DreamEventSink",
    "DreamReport",
    "DreamSignal",
    "EpisodicConsolidator",
    "FactCandidate",
    "FileDreamInbox",
    "InboxCounts",
    "MemoryAction",
    "MemoryObservation",
    "ReconcileOp",
    "RecurringSignalSkills",
    "SemanticConsolidator",
    "SessionSummarizer",
    "SessionSummary",
    "SkillCandidate",
    "SkillConsolidator",
    "TurnRecorder",
    "change_tree_lines",
]
