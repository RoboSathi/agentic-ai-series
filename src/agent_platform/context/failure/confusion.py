"""Confusion: too many near-identical options to choose between.

In the window: chunks that are near-copies of each other. In the request: more tools than a
model chooses well among, or tool schemas that crowd out everything else — checked where the
request is built (`llm/prompt.py`), since the window never sees the tools.
"""

from agent_platform.config import get_settings
from agent_platform.context.failure.common import finding
from agent_platform.context.similarity import jaccard
from agent_platform.context.types import AssembledContext, ContextFinding


def near_duplicate_ratio(context: AssembledContext) -> float:
    """Share of chunks that are near-copies of another chunk."""
    overlap = get_settings().context.consolidation.near_duplicate_overlap
    chunks = context.chunks
    if len(chunks) < 2:
        return 0.0
    duplicated = sum(
        any(i != j and jaccard(a.content, b.content) > overlap for j, b in enumerate(chunks))
        for i, a in enumerate(chunks)
    )
    return duplicated / len(chunks)


def window_findings(context: AssembledContext) -> list[ContextFinding]:
    ratio = get_settings().context.failure_detection.mode_ratio
    if near_duplicate_ratio(context) <= ratio:
        return []
    return [finding("confusion", "confusion_window", source="window", ratio=ratio)]


def tool_overload(tools: int, schema_tokens: int, total_tokens: int) -> ContextFinding | None:
    """A request offering more tools, or more schema, than the model chooses well among."""
    settings = get_settings().context.failure_detection
    share = schema_tokens / total_tokens if total_tokens else 0.0
    if tools <= settings.max_tools and share <= settings.max_tool_schema_share:
        return None
    return finding("confusion", "confusion_tools", source="request", tools=tools, share=share)
