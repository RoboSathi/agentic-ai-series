"""Offload: a bulky passage leaves the window, and the model can fetch it back.

A placeholder that says "fetch if needed" and offers no way to fetch is a deletion with a
polite note. Here an offloaded chunk's text is kept in the session's working memory under a
short handle, the placeholder names the handle and the tool, and `offload_fetch_tool` is that
tool: read-only, scoped to the session that offloaded it, offered only to an agent whose
window offloads.
"""

import hashlib
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from agent_platform.config import get_settings
from agent_platform.logging.trace_context import get_current_session
from agent_platform.memory.working import WorkingMemoryStore
from agent_platform.tools.types import ToolDefinition, ToolResult


@runtime_checkable
class OffloadStore(Protocol):
    """Where offloaded passages wait, by session and handle."""

    def put(self, session_id: str, text: str) -> str: ...

    def get(self, session_id: str, handle: str) -> str | None: ...


class WorkingMemoryOffloadStore:
    """Offloaded passages kept in the session's working memory (and so its checkpoints)."""

    def __init__(self, working_memory: WorkingMemoryStore) -> None:
        self._working_memory = working_memory

    def put(self, session_id: str, text: str) -> str:
        settings = get_settings().context.offload
        handle = hashlib.sha256(text.encode("utf-8")).hexdigest()[: settings.handle_chars]
        self._working_memory.get(session_id).offload(handle, text, settings.max_entries)
        return handle

    def get(self, session_id: str, handle: str) -> str | None:
        return self._working_memory.get(session_id).offloaded.get(handle.strip())


class FetchOffloadedArgs(BaseModel):
    # The handle the offloaded placeholder names.
    handle: str


def offload_tool_name() -> str:
    """The fetch tool's qualified name, as the registry and the offered set know it."""
    settings = get_settings().context.offload
    return f"{settings.tool_namespace}.{settings.tool_name}"


def offload_fetch_tool(store: OffloadStore) -> ToolDefinition:
    """The read-only tool that returns an offloaded passage of the caller's own session."""
    settings = get_settings().context.offload

    def fetch(handle: str) -> ToolResult:
        text = store.get(get_current_session() or "", handle)
        if text is None:
            return ToolResult(success=False, error=settings.not_found.format(handle=handle))
        return ToolResult(success=True, output=text, data={"handle": handle})

    return ToolDefinition(
        name=settings.tool_name,
        namespace=settings.tool_namespace,
        description=settings.tool_description,
        args_schema=FetchOffloadedArgs,
        handler=fetch,
        required_permission=settings.tool_permission or None,
        surface="editable",
        reversible=True,
        side_effecting=False,
    )


__all__ = [
    "FetchOffloadedArgs",
    "OffloadStore",
    "WorkingMemoryOffloadStore",
    "offload_fetch_tool",
    "offload_tool_name",
]
