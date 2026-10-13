"""One open MCP session, owned by a task of its own.

The SDK's client is an async context whose cancel scopes belong to the task that
entered it, and an agent's tool calls arrive from whichever task is running the
turn. So one task opens the client and keeps it, and callers hand it requests
through a queue (Active Object). A call fails — never hangs — when the server
has gone away, and closing the session ends the server process with it.

Every protocol event is reported to the session's `McpEventSink` (`events.py`),
which logs it in full to `mcp.log`: the session opening, the handshake (how it
was negotiated, what the server says it is, its instructions and capabilities,
how long it took), each tool listed, each call (arguments, result, error, time),
every notification the server sends (its log messages, a changed tool list,
progress), and the close — clean or not.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

from mcp import Client
from mcp.shared.dispatcher import ProgressFnT
from mcp.types import Implementation, LoggingMessageNotificationParams
from pydantic import BaseModel

from agent_platform.config import get_settings
from agent_platform.knowledge.mcp.client.events import (
    TOOLS_LIST_CHANGED,
    LoggedMcpEvents,
    McpEventSink,
)
from agent_platform.knowledge.mcp.common.types import McpCallResult, McpServerInfo, McpToolInfo
from agent_platform.logging import get_logger
from agent_platform.logging.trace_context import get_current_trace_id, set_current_trace_id

Request = Callable[[Client], Awaitable[Any]]
# One queued request: what to send, where its answer goes, and the trace of the turn that sent it.
Queued = tuple[Request, "asyncio.Future[Any]", str | None]
# Notifications the session reports through their own callbacks, not as raw messages.
_OWN_CALLBACKS = frozenset({"notifications/message", "notifications/progress"})
_MS = 1000


class McpSessionClosedError(RuntimeError):
    """The session (or the server behind it) is gone."""


class SdkMcpSession:
    """An `McpSession` over the MCP SDK's `Client`, served by its own task."""

    def __init__(
        self,
        target: Any,
        timeout_seconds: float,
        *,
        server: str = "",
        events: McpEventSink | None = None,
    ) -> None:
        self._target = target
        self._timeout = timeout_seconds
        self._server = server
        self._events = events or LoggedMcpEvents()
        self._requests: asyncio.Queue[Queued | None] = asyncio.Queue()
        # The trace of the request on the wire: a notification the server sends while answering
        # it is logged under the turn that asked, not the turn that opened the session.
        self._active_trace: str | None = None
        self._ready: asyncio.Future[None] | None = None
        self._task: asyncio.Task[None] | None = None
        self._info = McpServerInfo()
        self._changed = False
        self._logger = get_logger("mcp")

    def server_info(self) -> McpServerInfo:
        """What the server said about itself in the handshake (empty before it)."""
        return self._info

    async def open(self) -> None:
        """Start the owning task and wait until the server has answered the handshake."""
        self._events.opening(self._server, str(self._target))
        loop = asyncio.get_running_loop()
        self._ready = loop.create_future()
        self._task = asyncio.create_task(self._serve(), name="mcp-session")
        await self._ready

    def tools_changed(self) -> bool:
        return self._changed

    async def list_tools(self) -> list[McpToolInfo]:
        listing = await self._submit(lambda client: client.list_tools())
        self._changed = False
        return [
            McpToolInfo(
                name=tool.name,
                title=tool.title or "",
                description=tool.description or "",
                input_schema=dict(tool.input_schema or {}),
                output_schema=dict(tool.output_schema or {}),
                annotations=_wire(tool.annotations),
                meta=_wire(tool.meta),
            )
            for tool in listing.tools
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> McpCallResult:
        self._logger.info("MCP tools/call", server=self._server, tool=name, arguments=arguments)
        started = time.perf_counter()
        progress = self._progress(name)
        try:
            result = await self._submit(
                lambda client: client.call_tool(name, arguments, progress_callback=progress)
            )
        except Exception as exc:
            elapsed = _elapsed(started)
            self._events.call_failed(self._server, name, arguments, repr(exc), elapsed)
            raise
        text = "\n".join(
            block.text for block in result.content if getattr(block, "type", "") == "text"
        )
        outcome = McpCallResult(
            is_error=bool(result.is_error),
            text=text,
            structured=dict(result.structured_content or {}),
            duration_ms=_elapsed(started),
        )
        log = self._logger.warning if outcome.is_error else self._logger.info
        log(
            "MCP tools/call result",
            server=self._server,
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
        await self._requests.put((request, future, get_current_trace_id()))
        return await future

    async def _serve(self) -> None:
        started = time.perf_counter()
        try:
            async with self._client() as client:
                self._info = _server_info(client, _elapsed(started))
                self._events.connected(self._server, self._info)
                self._signal_ready(None)
                await self._loop(client)
            self._events.closed(self._server)
        except Exception as exc:  # noqa: BLE001 - every waiter is told, then it is reported
            self._events.session_failed(self._server, repr(exc))
            self._signal_ready(exc)
            self._fail_pending(exc)

    def _client(self) -> Client:
        """The SDK client, with every server-to-client channel reported."""
        settings = get_settings().mcp.client
        return Client(
            self._target,
            read_timeout_seconds=self._timeout,
            client_info=Implementation(name=settings.name, version=settings.version),
            mode=settings.mode,
            log_level=settings.log_level,
            logging_callback=self._on_log,
            message_handler=self._on_message,
        )

    async def _on_log(self, params: LoggingMessageNotificationParams) -> None:
        set_current_trace_id(self._active_trace)
        self._events.server_log(self._server, str(params.level), params.logger or "", params.data)

    async def _on_message(self, message: Any) -> None:
        """A raw server notification (or a transport error) the other callbacks do not take."""
        set_current_trace_id(self._active_trace)
        if isinstance(message, Exception):
            self._events.notification(
                self._server, type(message).__name__, {"error": repr(message)}
            )
            return
        method = str(getattr(message, "method", ""))
        if method in _OWN_CALLBACKS:
            return
        params: Any = getattr(message, "params", None)
        shown = params.model_dump(mode="json") if isinstance(params, BaseModel) else {}
        self._changed = self._changed or method == TOOLS_LIST_CHANGED
        self._events.notification(self._server, method, shown)

    def _progress(self, tool: str) -> ProgressFnT:
        async def report(progress: float, total: float | None, message: str | None) -> None:
            set_current_trace_id(self._active_trace)
            self._events.progress(self._server, tool, progress, total, message or "")

        return report

    async def _loop(self, client: Client) -> None:
        while (item := await self._requests.get()) is not None:
            request, future, trace = item
            self._active_trace = trace
            set_current_trace_id(trace)
            try:
                future.set_result(await request(client))
            except Exception as exc:  # noqa: BLE001 - handed to the caller
                future.set_exception(exc)
            finally:
                self._active_trace = None

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


def _server_info(client: Client, duration_ms: float) -> McpServerInfo:
    """The handshake's answer: the server's name and version, instructions, capabilities —
    and how it was negotiated (`server/discover` from `mcp.client.discover_from` on, else
    `initialize`), what this client declared, and how long it took."""
    implementation = getattr(client, "server_info", None)
    capabilities = _wire(getattr(client, "server_capabilities", None))
    declared = list(capabilities)
    settings = get_settings().mcp.client
    protocol = str(getattr(client, "protocol_version", "") or "")
    discovered = bool(protocol) and protocol >= settings.discover_from
    return McpServerInfo(
        name=str(getattr(implementation, "name", "") or ""),
        version=str(getattr(implementation, "version", "") or ""),
        protocol_version=protocol,
        instructions=str(getattr(client, "instructions", "") or ""),
        capabilities=declared,
        server_capabilities=capabilities,
        client_capabilities=_client_capabilities(client, protocol),
        handshake=settings.discover_method if discovered else settings.initialize_method,
        client=f"{settings.name} {settings.version}",
        duration_ms=duration_ms,
    )


def _client_capabilities(client: Client, protocol: str) -> dict[str, Any]:
    """What this client declared in the handshake, as it went on the wire.

    The SDK builds the declaration from the callbacks it was given and keeps no copy; it is
    rebuilt the same way for the version negotiated. Nothing is guessed when it cannot be.
    """
    build = getattr(client.session, "_build_capabilities", None)
    if build is None or not protocol:
        return {}
    return _wire(build(protocol))


def _wire(model: Any) -> dict[str, Any]:
    """A protocol object as it reads on the wire (camelCase names, unset fields left out)."""
    if not isinstance(model, BaseModel):
        return dict(model) if isinstance(model, dict) else {}
    dumped: dict[str, Any] = model.model_dump(by_alias=True, mode="json", exclude_none=True)
    return dumped


def _elapsed(started: float) -> float:
    return round((time.perf_counter() - started) * _MS, 2)


__all__ = ["McpSessionClosedError", "SdkMcpSession"]
