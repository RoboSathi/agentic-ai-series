"""Memory dreams: consolidating the dream inbox into long-term memory (`make dream`)."""

from agent_platform.memory.consolidation.report import (
    DreamEntry,
    DreamReport,
    ReconcileOp,
    change_tree_lines,
)
from agent_platform.memory.consolidation.service import ConsolidationService
from agent_platform.memory.consolidation.summaries import DialogueSummarizer, SessionSummarizer

__all__ = [
    "ConsolidationService",
    "DialogueSummarizer",
    "DreamEntry",
    "DreamReport",
    "ReconcileOp",
    "SessionSummarizer",
    "change_tree_lines",
]
