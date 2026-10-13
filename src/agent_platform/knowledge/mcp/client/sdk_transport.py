"""Reaching an MCP server: stdio (launch the process) or HTTP (call the URL).

Which servers exist is an agent's configuration (`mcp.servers`); this turns one
entry into something the SDK can connect to and opens a session on it. A stdio
server is started in the repository root with this interpreter (the command's
`{python}`), and inherits the environment, so it reads the same configuration
as the agent that launched it.
"""

import os
import sys
from collections.abc import Callable
from typing import Any

from mcp import StdioServerParameters

from agent_platform.config import get_settings
from agent_platform.config.paths import REPO_ROOT
from agent_platform.config.settings import McpServerSettings
from agent_platform.knowledge.mcp.client.contracts import McpServerSpec
from agent_platform.knowledge.mcp.client.events import McpEventSink
from agent_platform.knowledge.mcp.client.sdk_session import SdkMcpSession

# A test (or an embedding) hands the SDK a target directly: an in-process server, say.
TargetResolver = Callable[[McpServerSpec], Any]


def server_spec(settings: McpServerSettings) -> McpServerSpec:
    """The configured server as the protocol contract names it."""
    return McpServerSpec(
        name=settings.name,
        transport=settings.transport,
        command=settings.command,
        url=settings.url,
        env=settings.env,
        namespace=settings.namespace,
    )


class SdkMcpTransport:
    """An `McpTransport` over the MCP SDK."""

    def __init__(self, resolve: TargetResolver | None = None) -> None:
        self._resolve = resolve or _target

    async def connect(
        self, server: McpServerSpec, events: McpEventSink | None = None
    ) -> SdkMcpSession:
        session = SdkMcpSession(
            self._resolve(server),
            get_settings().mcp.call_timeout_seconds,
            server=server.name,
            events=events,
        )
        await session.open()
        return session


def _target(server: McpServerSpec) -> Any:
    if server.transport == "http":
        return server.url
    placeholder = get_settings().mcp.python_placeholder
    command = [sys.executable if part == placeholder else part for part in server.command]
    return StdioServerParameters(
        command=command[0],
        args=command[1:],
        env={**os.environ, **server.env},
        cwd=str(REPO_ROOT),
    )


__all__ = ["SdkMcpTransport", "TargetResolver", "server_spec"]
