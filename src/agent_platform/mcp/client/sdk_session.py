"""One open MCP session, owned by a task of its own.

The SDK's client is an async context whose cancel scopes belong to the task that
entered it, and an agent's tool calls arrive from whichever task is running the
turn. So one task opens the client and keeps it, and callers hand it requests
through a queue (Active Object). A call fails — never hangs — when the server
has gone away, and closing the session ends the server process with it.

Every protocol event is logged in full (`mcp.log`): the handshake (what the server
says it is, its instructions and capabilities), each tool it lists (description and
input schema), and each call (arguments, result, error, time).
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

from mcp import Client

from agent_platform.logging import get_logger
from agent_platform.mcp.common.types import McpCallResult, McpServerInfo, McpToolInfo

Request = Callable[[Client], Awaitable[Any]]


class McpSessionClosedError(RuntimeError):
    """The session (or the server behind it) is gone."""


class SdkMcpSession:
    """An `McpSession` over the MCP SDK's `Client`, served by its own task."""

    def __init__(self, target: Any, timeout_seconds: float) -> None:
        self._target = target
        self._timeout = timeout_seconds
        self._requests: asyncio.Queue[tuple[Request, asyncio.Future[Any]] | None] = asyncio.Queue()
        self._ready: asyncio.Future[None] | None = None
        self._task: asyncio.Task[None] | None = None
        self._info = McpServerInfo()
        self._logger = get_logger("mcp")

    def server_info(self) -> McpServerInfo:
        """What the server said about itself in the handshake (empty before it)."""
        return self._info

    async def open(self) -> None:
        """Start the owning task and wait until the server has answered the handshake."""
        loop = asyncio.get_running_loop()
        self._ready = loop.create_future()
        self._task = asyncio.create_task(self._serve(), name="mcp-session")
        await self._ready

    async def list_tools(self) -> list[McpToolInfo]:
        listing = await self._submit(lambda client: client.list_tools())
        tools = [
            McpToolInfo(
                name=tool.name,
                description=tool.description or "",
                input_schema=dict(tool.input_schema or {}),
            )
            for tool in listing.tools
        ]
        self._logger.info(
            "MCP tools/list",
            server=self._info.name,
            tools=[tool.model_dump() for tool in tools],
        )
        return tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> McpCallResult:
        self._logger.info("MCP tools/call", server=self._info.name, tool=name, arguments=arguments)
        started = time.perf_counter()
        try:
            result = await self._submit(lambda client: client.call_tool(name, arguments))
        except Exception as exc:
            elapsed = round((time.perf_counter() - started) * 1000, 2)
            self._logger.warning(
                "MCP tools/call failed", tool=name, error=repr(exc), duration_ms=elapsed
            )
            raise
        text = "\n".join(
            block.text for block in result.content if getattr(block, "type", "") == "text"
        )
        outcome = McpCallResult(
            is_error=bool(result.is_error),
            text=text,
            structured=dict(result.structured_content or {}),
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        log = self._logger.warning if outcome.is_error else self._logger.info
        log(
            "MCP tools/call result",
            tool=name,
            is_error=outcome.is_error,
            duration_ms=outcome.duration_ms,
            text=outcome.text,
            structured=outcome.structured,
        )
        return outcome

    async def close(self) -> None:
        if self._task is None or self._task.done():
            return
        await self._requests.put(None)
        await self._task

    async def _submit(self, request: Request) -> Any:
        if self._task is None or self._task.done():
            raise McpSessionClosedError("the MCP session is not open")
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        await self._requests.put((request, future))
        return await future

    async def _serve(self) -> None:
        try:
            async with Client(self._target, read_timeout_seconds=self._timeout) as client:
                self._info = _server_info(client)
                self._logger.info("MCP server connected", **self._info.model_dump())
                self._signal_ready(None)
                await self._loop(client)
        except Exception as exc:  # noqa: BLE001 - every waiter is told, then it is logged
            self._logger.warning("MCP session ended", error=repr(exc))
            self._signal_ready(exc)
            self._fail_pending(exc)

    async def _loop(self, client: Client) -> None:
        while (item := await self._requests.get()) is not None:
            request, future = item
            try:
                future.set_result(await request(client))
            except Exception as exc:  # noqa: BLE001 - handed to the caller
                future.set_exception(exc)

    def _signal_ready(self, error: BaseException | None) -> None:
        if self._ready is None or self._ready.done():
            return
        if error is None:
            self._ready.set_result(None)
        else:
            self._ready.set_exception(McpSessionClosedError(str(error)))

    def _fail_pending(self, error: BaseException) -> None:
        while not self._requests.empty():
            item = self._requests.get_nowait()
            if item is not None and not item[1].done():
                item[1].set_exception(McpSessionClosedError(str(error)))


def _server_info(client: Client) -> McpServerInfo:
    """The handshake's answer: the server's name and version, instructions, capabilities."""
    implementation = getattr(client, "server_info", None)
    capabilities = getattr(client, "server_capabilities", None)
    declared = (
        [name for name, value in capabilities.model_dump().items() if value]
        if capabilities is not None
        else []
    )
    return McpServerInfo(
        name=str(getattr(implementation, "name", "") or ""),
        version=str(getattr(implementation, "version", "") or ""),
        protocol_version=str(getattr(client, "protocol_version", "") or ""),
        instructions=str(getattr(client, "instructions", "") or ""),
        capabilities=declared,
    )


__all__ = ["McpSessionClosedError", "SdkMcpSession"]
