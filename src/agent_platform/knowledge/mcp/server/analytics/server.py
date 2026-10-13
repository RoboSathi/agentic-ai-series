"""The analytics source over the Model Context Protocol: five read-only tools and a manifest.

`list_metrics` → `compute_metric` answers a question about a catalogue metric with the
catalogue's own formula (ADR 0009), so no model re-derives "resolution rate" its own
way. `list_views` → `describe_view` → `run_query` is the rest: discover what exists,
learn what its columns mean, ask one SELECT. Nothing here
writes; everything a query may touch is the sealed `SourceDatabase`. A rejected
query comes back as a tool error whose message says what to change, so the
model can correct itself instead of guessing. The server also says it over the
protocol: a refusal is a `notifications/message` warning, a result to double-check
(no rows, rows cut) an info message — to a client that opted in to its logs — and
every request and refusal is in the server's own log file.

Any MCP client can connect — the platform's own ops agent, Claude Code, the MCP
Inspector — over stdio: `python -m agent_platform.knowledge.mcp.server.analytics`.
"""

import asyncio
import warnings
from collections.abc import Callable
from typing import TypeVar

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.shared.exceptions import MCPDeprecationWarning
from mcp_types import LoggingLevel, ToolAnnotations
from pydantic import BaseModel

from agent_platform.analytics.query_guard import QueryRejectedError
from agent_platform.analytics.source_db import (
    MetricSummary,
    QueryResult,
    SourceDatabase,
    ViewDescription,
    ViewSummary,
)
from agent_platform.config import get_settings
from agent_platform.logging import get_logger

T = TypeVar("T")
_WARNING: LoggingLevel = "warning"
_INFO: LoggingLevel = "info"
READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False)


class ViewList(BaseModel):
    """`list_views`: the curated views (its outputSchema)."""

    views: list[ViewSummary]


class MetricCatalogue(BaseModel):
    """`list_metrics`: the catalogue, and the agent ids the source holds data for."""

    metrics: list[MetricSummary]
    agents: list[str]


def build_server(database: SourceDatabase | None = None) -> MCPServer:
    """The analytics MCP server over `database` (the configured source by default)."""
    settings = get_settings().analytics.mcp_server
    source = database or SourceDatabase()
    server = MCPServer(
        name=settings.name, version=settings.version, instructions=settings.instructions
    )

    @server.tool(annotations=READ_ONLY)
    async def list_views() -> ViewList:
        """Every curated view in the analytics source: what it holds and how many rows."""
        return ViewList(views=await asyncio.to_thread(source.views))

    @server.tool(annotations=READ_ONLY)
    async def describe_view(name: str, ctx: Context) -> ViewDescription:
        """One view's columns (name, type, meaning) and a few sample rows. Call before querying it."""
        return await _answer(ctx, source.describe, name)

    @server.tool(annotations=READ_ONLY)
    async def list_metrics() -> MetricCatalogue:
        """The metric catalogue: every metric's meaning, direction, breakdowns and SLO target,
        and the agent ids the source holds data for (compute_metric's agent_id is one of them).

        Tier 1 metrics hold for every agent; tier 2 metrics belong to the agent named as owner.
        """
        metrics = await asyncio.to_thread(source.metrics)
        return MetricCatalogue(metrics=metrics, agents=await asyncio.to_thread(source.agents))

    @server.tool(annotations=READ_ONLY)
    async def compute_metric(
        ctx: Context,
        name: str,
        agent_id: str | None = None,
        days: int | None = None,
        by: str | None = None,
    ) -> QueryResult:
        """One catalogue metric by its own formula, per agent (runtime traffic, evals excluded).

        agent_id narrows to one agent; days to the last N days (omit for all time); by to
        one of the metric's breakdowns (see list_metrics). Prefer this to re-deriving a metric.
        """
        arguments = (name, agent_id, days, by)
        return await _answer(ctx, lambda _: source.compute(*arguments), name)

    @server.tool(annotations=READ_ONLY)
    async def run_query(sql: str, ctx: Context) -> QueryResult:
        """Run ONE read-only SELECT over the curated views (DuckDB SQL). Rows are capped."""
        return await _answer(ctx, source.query, sql)

    @server.resource(settings.manifest_uri, mime_type="application/json")
    def manifest() -> str:
        """When the source was exported, and each view's row count."""
        current = source.manifest
        return current.model_dump_json() if current is not None else "{}"

    return server


async def _answer(ctx: Context, call: Callable[[str], T], argument: str) -> T:
    """Run a blocking source call off the loop; a refusal becomes a tool error the model reads.

    Logged on this side (the file log), and told to the client over the protocol.
    """
    logger = get_logger("mcp")
    logger.info("MCP server request", request_id=ctx.request_id, argument=argument)
    try:
        answer = await asyncio.to_thread(call, argument)
    except QueryRejectedError as exc:
        logger.warning("MCP server refused", request_id=ctx.request_id, reason=str(exc))
        await _tell(ctx, _WARNING, str(exc))
        raise ToolError(str(exc)) from exc
    for note in getattr(answer, "notes", []):
        await _tell(ctx, _INFO, note)
    return answer


async def _tell(ctx: Context, level: LoggingLevel, message: str) -> None:
    """One `notifications/message` to the client (sent only to one that opted in to logs).

    The logging capability is deprecated from protocol 2026-07-28, but still delivered on
    request: it is how a client sees what the server thought, beside what it returned.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MCPDeprecationWarning)
        await ctx.log(level, message, logger_name=get_settings().analytics.mcp_server.name)


__all__ = ["build_server"]
