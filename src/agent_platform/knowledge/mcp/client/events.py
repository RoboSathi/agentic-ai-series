"""Every MCP protocol event, logged in full; the ones an operator acts on, shown as cards.

A session reports what happens on the wire to an `McpEventSink`: opening, the
handshake, the tool listing, each call (answered or not), every notification the
server sends (its log messages, a changed tool list, progress), and the close.
`LoggedMcpEvents` writes each to `mcp.log` with everything it carried, so the log
is the protocol's full record. `McpEvents` adds the live feed, which shows only
what changes what an operator does next:

    MCP SERVER        the handshake and the tools offered, once per connection;
                      or the connection failing
    MCP CALL          a call that got no answer (one that did is drawn by the
                      step's own cards, `StepEvents.mcp_call`)
    MCP NOTIFICATION  the server changed its tools, or logged a warning or worse
    MCP SESSION       the session ended on an error

Progress, server logs below the card level, and a clean close stay in the log.
Any agent that consumes MCP gets these from the platform, unchanged.
"""

from typing import Any, Protocol, runtime_checkable

from agent_platform.config import get_settings
from agent_platform.knowledge.mcp.client.contracts import McpServerDescription
from agent_platform.knowledge.mcp.client.presentation import TOOLS_CALL, server_card
from agent_platform.knowledge.mcp.common.types import McpServerInfo
from agent_platform.logging import get_logger
from agent_platform.observability.status_feed import StageReporter
from agent_platform.observability.step_events import MCP_CALL
from agent_platform.observability.types import StatusKind

# The feed's stage keys for the protocol's own cards (`tui/labels.py` titles them).
MCP_SERVER = "mcp_server"
MCP_NOTIFICATION = "mcp_notification"
MCP_SESSION = "mcp_session"
# Notification methods the client acts on.
TOOLS_LIST_CHANGED = "notifications/tools/list_changed"


@runtime_checkable
class McpEventSink(Protocol):
    """Where a session reports what happened on the wire."""

    def opening(self, server: str, target: str) -> None: ...

    def connected(self, server: str, info: McpServerInfo) -> None: ...

    def connect_failed(self, server: str, error: str) -> None: ...

    def tools_listed(self, description: McpServerDescription) -> None: ...

    def call_failed(
        self, server: str, tool: str, arguments: dict[str, Any], error: str, duration_ms: float
    ) -> None: ...

    def notification(self, server: str, method: str, params: dict[str, Any]) -> None: ...

    def server_log(self, server: str, level: str, logger: str, data: Any) -> None: ...

    def progress(
        self, server: str, tool: str, progress: float, total: float | None, message: str
    ) -> None: ...

    def closed(self, server: str) -> None: ...

    def session_failed(self, server: str, error: str) -> None: ...


class LoggedMcpEvents:
    """Every event to `mcp.log`, in full — the record whether or not anyone is watching."""

    def __init__(self) -> None:
        self._logger = get_logger("mcp")

    def opening(self, server: str, target: str) -> None:
        self._logger.info("MCP session opening", server=server, target=target)

    def connected(self, server: str, info: McpServerInfo) -> None:
        self._logger.info("MCP server connected", server=server, **info.model_dump())

    def connect_failed(self, server: str, error: str) -> None:
        self._logger.warning("MCP connection failed", server=server, error=error)

    def tools_listed(self, description: McpServerDescription) -> None:
        self._logger.info(
            "MCP tools/list",
            server=description.server.name,
            tools=[tool.model_dump() for tool in description.tools],
        )

    def call_failed(
        self, server: str, tool: str, arguments: dict[str, Any], error: str, duration_ms: float
    ) -> None:
        self._logger.warning(
            "MCP tools/call failed",
            server=server,
            tool=tool,
            arguments=arguments,
            error=error,
            duration_ms=duration_ms,
        )

    def notification(self, server: str, method: str, params: dict[str, Any]) -> None:
        self._logger.info("MCP notification", server=server, method=method, params=params)

    def server_log(self, server: str, level: str, logger: str, data: Any) -> None:
        self._logger.info(
            "MCP server log", server=server, server_level=level, server_logger=logger, data=data
        )

    def progress(
        self, server: str, tool: str, progress: float, total: float | None, message: str
    ) -> None:
        self._logger.info(
            "MCP progress",
            server=server,
            tool=tool,
            progress=progress,
            total=total,
            progress_message=message,
        )

    def closed(self, server: str) -> None:
        self._logger.info("MCP session closed", server=server)

    def session_failed(self, server: str, error: str) -> None:
        self._logger.warning("MCP session ended", server=server, error=error)


class McpEvents(LoggedMcpEvents):
    """The log, plus a card for each event an operator acts on (`mcp.card`)."""

    def __init__(self, reporter: StageReporter | None) -> None:
        super().__init__()
        self._reporter = reporter

    def connect_failed(self, server: str, error: str) -> None:
        super().connect_failed(server, error)
        heading = _words().connect_failed_heading.format(server=server, error=error)
        self._emit(MCP_SERVER, "failed", heading, {_words().error_title: [error]})

    def tools_listed(self, description: McpServerDescription) -> None:
        super().tools_listed(description)
        heading, sections = server_card(description)
        self._emit(MCP_SERVER, "completed", heading, sections)

    def call_failed(
        self, server: str, tool: str, arguments: dict[str, Any], error: str, duration_ms: float
    ) -> None:
        super().call_failed(server, tool, arguments, error, duration_ms)
        words = _words()
        heading = words.call_raised_heading.format(
            server=server, method=TOOLS_CALL, tool=tool, duration_ms=duration_ms
        )
        self._emit(MCP_CALL, "failed", heading, {words.error_title: [error]})

    def notification(self, server: str, method: str, params: dict[str, Any]) -> None:
        super().notification(server, method, params)
        if method == TOOLS_LIST_CHANGED:
            heading = _words().tools_changed_heading.format(server=server, method=method)
            self._emit(MCP_NOTIFICATION, "warning", heading, {})

    def server_log(self, server: str, level: str, logger: str, data: Any) -> None:
        super().server_log(server, level, logger, data)
        settings = get_settings().mcp.client
        if level not in settings.card_log_levels:
            return
        lines = (data if isinstance(data, str) else str(data)).splitlines() or [""]
        # The header is the message's first line; the card's body is all of it.
        heading = _words().server_log_heading.format(server=server, level=level, message=lines[0])
        status: StatusKind = "failed" if level in settings.failed_log_levels else "warning"
        self._emit(MCP_NOTIFICATION, status, heading, {_words().message_title: lines})

    def session_failed(self, server: str, error: str) -> None:
        super().session_failed(server, error)
        heading = _words().session_failed_heading.format(server=server, error=error)
        self._emit(MCP_SESSION, "failed", heading, {_words().error_title: [error]})

    def _emit(
        self, stage: str, status: StatusKind, heading: str, sections: dict[str, list[str]]
    ) -> None:
        if self._reporter is None:
            return
        self._reporter.emit(stage, status, event_kind="narrative", summary=heading, fields=sections)


def _words() -> Any:
    return get_settings().mcp.card


__all__ = [
    "MCP_CALL",
    "MCP_NOTIFICATION",
    "MCP_SERVER",
    "MCP_SESSION",
    "TOOLS_LIST_CHANGED",
    "LoggedMcpEvents",
    "McpEventSink",
    "McpEvents",
]
