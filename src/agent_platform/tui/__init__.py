"""The terminal UI — a consumer of observability, never a source of truth (LLD §20)."""

from agent_platform.tui.app import AgenticTuiApp, main
from agent_platform.tui.clipboard import ClipboardService, Osc52ClipboardService
from agent_platform.tui.labels import StageLabeler, StageLabelRegistry
from agent_platform.tui.widgets import (
    LiveStatusPanel,
    StageEventWidget,
    StatusLineWidget,
    TokenCounterWidget,
)

__all__ = [
    "AgenticTuiApp",
    "ClipboardService",
    "LiveStatusPanel",
    "Osc52ClipboardService",
    "StageLabelRegistry",
    "StageLabeler",
    "StatusLineWidget",
    "TokenCounterWidget",
    "StageEventWidget",
    "main",
]
