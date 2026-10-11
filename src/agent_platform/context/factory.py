"""Context management for one agent, composed once in its composition root (Factory).

Which strategies run, whether the window is checked, and whether the offload fetch tool
exists all follow from the agent's declared features (`context`, `context.failure_detection`)
and the platform's `context.*` settings. Read here, once; the pieces never consult config.
"""

from dataclasses import dataclass

from agent_platform.config import get_settings
from agent_platform.config.features import FeatureSet
from agent_platform.context.failure import (
    ContextFailureDetector,
    ContradictionModel,
    InjectionScreen,
)
from agent_platform.context.offload import WorkingMemoryOffloadStore, offload_fetch_tool
from agent_platform.context.strategies import StrategySelector
from agent_platform.memory.working import WorkingMemoryStore
from agent_platform.tools.types import ToolDefinition

CONTEXT = "context"
FAILURE_DETECTION = "context.failure_detection"


@dataclass(frozen=True)
class ContextManagement:
    """The window's strategies, its health check, and the tool that fetches offloaded text."""

    strategies: StrategySelector | None
    health: ContextFailureDetector | None
    fetch_tool: ToolDefinition | None


def build_context_management(
    features: FeatureSet,
    working_memory: WorkingMemoryStore | None,
    screen: InjectionScreen | None = None,
) -> ContextManagement:
    """Everything a declared `context` feature needs; nothing when it is withheld."""
    if not features.is_on(CONTEXT):
        return ContextManagement(None, None, None)
    store = WorkingMemoryOffloadStore(working_memory) if working_memory is not None else None
    strategies = StrategySelector(offload_store=store)
    health = (
        ContextFailureDetector(screen=screen, nli=_nli())
        if features.is_on(FAILURE_DETECTION)
        else None
    )
    tool = offload_fetch_tool(store) if store is not None and strategies.offloads else None
    return ContextManagement(strategies, health, tool)


def _nli() -> ContradictionModel | None:
    """The local NLI model the prose checks ask, when they are switched on."""
    if not get_settings().context.failure_detection.nli:
        return None
    from agent_platform.llm.local_nli import shared_local_nli_model

    return shared_local_nli_model()


__all__ = ["ContextManagement", "build_context_management"]
