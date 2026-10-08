"""One open MCP session, owned by a task of its own.

The SDK's client is an async context whose cancel scopes belong to the task that
entered it, and an agent's tool calls arrive from whichever task is running the
turn. So one task opens the client and keeps it, and callers hand it requests
through a queue (Active Object). A call fails — never hangs — when the server
has gone away, and closing the session ends the server process with it.
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from mcp import Client

from agent_platform.logging import get_logger
from agent_platform.mcp.types import McpCallResult, McpToolInfo

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
        self._logger = get_logger("mcp")

    async def open(self) -> None:
        """Start the owning task and wait until the server has answered the handshake."""
        loop = asyncio.get_running_loop()
        self._ready = loop.create_future()
        self._task = asyncio.create_task(self._serve(), name="mcp-session")
        await self._ready

    async def list_tools(self) -> list[McpToolInfo]:
        listing = await self._submit(lambda client: client.list_tools())
        return [
            McpToolInfo(
                name=tool.name,
                description=tool.description or "",
                input_schema=dict(tool.input_schema or {}),
            )
            for tool in listing.tools
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> McpCallResult:
        result = await self._submit(lambda client: client.call_tool(name, arguments))
        text = "\n".join(
            block.text for block in result.content if getattr(block, "type", "") == "text"
        )
        return McpCallResult(
            is_error=bool(result.is_error),
            text=text,
            structured=dict(result.structured_content or {}),
        )

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


__all__ = ["McpSessionClosedError", "SdkMcpSession"]
