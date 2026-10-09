"""`python -m agent_platform.analytics.mcp_server` — serve the source over stdio.

On stdio, stdout *is* the protocol, so console logging is switched off before
settings load; everything is still logged to the files under `logging.dir`.
"""

import os

os.environ["LOGGING__CONSOLE_ENABLED"] = "false"

from agent_platform.analytics.mcp_server.server import build_server  # noqa: E402

build_server().run("stdio")
