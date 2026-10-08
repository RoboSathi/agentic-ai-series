"""The analytics mart over the Model Context Protocol: three read-only tools and a manifest.

`list_views` → `describe_view` → `run_query` is the whole protocol a model needs:
discover what exists, learn what its columns mean, ask one SELECT. Nothing here
writes; everything a query may touch is the sealed `MartDatabase`. A rejected
query comes back as a tool error whose message says what to change, so the
model can correct itself instead of guessing.

Any MCP client can connect — the platform's own ops agent, Claude Code, the MCP
Inspector — over stdio: `python -m agent_platform.analytics.mcp_server`.
"""

import asyncio
from collections.abc import Callable
from typing import Any, TypeVar

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations

from agent_platform.analytics.mart_db import MartDatabase
from agent_platform.analytics.query_guard import QueryRejectedError
from agent_platform.config import get_settings

T = TypeVar("T")
READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False)


def build_server(database: MartDatabase | None = None) -> MCPServer:
    """The analytics MCP server over `database` (the configured mart by default)."""
    settings = get_settings().analytics.mcp_server
    mart = database or MartDatabase()
    server = MCPServer(name=settings.name, instructions=settings.instructions)

    @server.tool(annotations=READ_ONLY)
    async def list_views() -> dict[str, Any]:
        """Every curated view in the analytics mart: what it holds and how many rows."""
        views = await asyncio.to_thread(mart.views)
        return {"views": [view.model_dump() for view in views]}

    @server.tool(annotations=READ_ONLY)
    async def describe_view(name: str) -> dict[str, Any]:
        """One view's columns (name, type, meaning) and a few sample rows. Call before querying it."""
        return (await _answer(mart.describe, name)).model_dump()

    @server.tool(annotations=READ_ONLY)
    async def run_query(sql: str) -> dict[str, Any]:
        """Run ONE read-only SELECT over the curated views (DuckDB SQL). Rows are capped."""
        return (await _answer(mart.query, sql)).model_dump()

    @server.resource(settings.manifest_uri, mime_type="application/json")
    def manifest() -> str:
        """When the mart was exported, and each view's row count."""
        current = mart.manifest
        return current.model_dump_json() if current is not None else "{}"

    return server


async def _answer(call: Callable[[str], T], argument: str) -> T:
    """Run a blocking mart call off the loop; a refusal becomes a tool error the model reads."""
    try:
        return await asyncio.to_thread(call, argument)
    except QueryRejectedError as exc:
        raise ToolError(str(exc)) from exc


__all__ = ["build_server"]
