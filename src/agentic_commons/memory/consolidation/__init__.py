"""Memory dreams: consolidating the dream inbox into long-term memory (`make dream`)."""

from agentic_commons.memory.consolidation.report import (
    DreamEntry,
    DreamReport,
    ReconcileOp,
    change_tree_lines,
)
from agentic_commons.memory.consolidation.service import ConsolidationService
from agentic_commons.memory.consolidation.summaries import DialogueSummarizer, SessionSummarizer

__all__ = [
    "ConsolidationService",
    "DialogueSummarizer",
    "DreamEntry",
    "DreamReport",
    "ReconcileOp",
    "SessionSummarizer",
    "change_tree_lines",
]
