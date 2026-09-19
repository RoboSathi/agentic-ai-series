"""The terminal UI — a consumer of observability, never a source of truth (LLD §20)."""

from agentic_commons.tui.app import AgenticTuiApp, main
from agentic_commons.tui.clipboard import ClipboardService, Osc52ClipboardService
from agentic_commons.tui.labels import StageLabeler, StageLabelRegistry
from agentic_commons.tui.widgets import (
    LiveStatusPanel,
    #StatusFeedBuffer,
    StatusLineWidget,
    TokenCounterWidget,
    StageEventWidget,
)

__all__ = [
    "AgenticTuiApp",
    "ClipboardService",
    "LiveStatusPanel",
    "Osc52ClipboardService",
    "StageLabelRegistry",
    "StageLabeler",
    "StatusFeedBuffer",
    "StatusLineWidget",
    "TokenCounterWidget",
    "StageEventWidget",
    "main",
]
