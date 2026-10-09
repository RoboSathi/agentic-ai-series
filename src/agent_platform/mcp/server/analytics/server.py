"""The analytics source over the Model Context Protocol: five read-only tools and a manifest.

`list_metrics` → `compute_metric` answers a question about a catalogue metric with the
catalogue's own formula (ADR 0009), so no model re-derives "resolution rate" its own
way. `list_views` → `describe_view` → `run_query` is the rest: discover what exists,
learn what its columns mean, ask one SELECT. Nothing here
writes; everything a query may touch is the sealed `SourceDatabase`. A rejected
query comes back as a tool error whose message says what to change, so the
model can correct itself instead of guessing.

Any MCP client can connect — the platform's own ops agent, Claude Code, the MCP
Inspector — over stdio: `python -m agent_platform.mcp.server.analytics`.
"""

import asyncio
from collections.abc import Callable
from typing import Any, TypeVar

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations

from agent_platform.analytics.query_guard import QueryRejectedError
from agent_platform.analytics.source_db import SourceDatabase
from agent_platform.config import get_settings

T = TypeVar("T")
READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False)


def build_server(database: SourceDatabase | None = None) -> MCPServer:
    """The analytics MCP server over `database` (the configured source by default)."""
    settings = get_settings().analytics.mcp_server
    source = database or SourceDatabase()
    server = MCPServer(name=settings.name, instructions=settings.instructions)

    @server.tool(annotations=READ_ONLY)
    async def list_views() -> dict[str, Any]:
        """Every curated view in the analytics source: what it holds and how many rows."""
        views = await asyncio.to_thread(source.views)
        return {"views": [view.model_dump() for view in views]}

    @server.tool(annotations=READ_ONLY)
    async def describe_view(name: str) -> dict[str, Any]:
        """One view's columns (name, type, meaning) and a few sample rows. Call before querying it."""
        return (await _answer(source.describe, name)).model_dump()

    @server.tool(annotations=READ_ONLY)
    async def list_metrics() -> dict[str, Any]:
        """The metric catalogue: every metric's meaning, direction, breakdowns and SLO target.

        Tier 1 metrics hold for every agent; tier 2 metrics belong to the agent named as owner.
        """
        metrics = await asyncio.to_thread(source.metrics)
        return {"metrics": [metric.model_dump() for metric in metrics]}

    @server.tool(annotations=READ_ONLY)
    async def compute_metric(
        name: str, agent_id: str | None = None, days: int | None = None, by: str | None = None
    ) -> dict[str, Any]:
        """One catalogue metric by its own formula, per agent (runtime traffic, evals excluded).

        agent_id narrows to one agent; days to the last N days (omit for all time); by to
        one of the metric's breakdowns (see list_metrics). Prefer this to re-deriving a metric.
        """
        try:
            result = await asyncio.to_thread(source.compute, name, agent_id, days, by)
        except QueryRejectedError as exc:
            raise ToolError(str(exc)) from exc
        return result.model_dump()

    @server.tool(annotations=READ_ONLY)
    async def run_query(sql: str) -> dict[str, Any]:
        """Run ONE read-only SELECT over the curated views (DuckDB SQL). Rows are capped."""
        return (await _answer(source.query, sql)).model_dump()

    @server.resource(settings.manifest_uri, mime_type="application/json")
    def manifest() -> str:
        """When the source was exported, and each view's row count."""
        current = source.manifest
        return current.model_dump_json() if current is not None else "{}"

    return server


async def _answer(call: Callable[[str], T], argument: str) -> T:
    """Run a blocking source call off the loop; a refusal becomes a tool error the model reads."""
    try:
        return await asyncio.to_thread(call, argument)
    except QueryRejectedError as exc:
        raise ToolError(str(exc)) from exc


__all__ = ["build_server"]
