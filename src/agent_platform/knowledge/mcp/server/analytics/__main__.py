"""`python -m agent_platform.knowledge.mcp.server.analytics` — serve the analytics source over stdio."""

from agent_platform.knowledge.mcp.common.stdio import silence_console

silence_console()

from agent_platform.knowledge.mcp.server.analytics.server import build_server  # noqa: E402

build_server().run("stdio")
