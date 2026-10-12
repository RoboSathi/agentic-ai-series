"""The ops agent — the platform's third product: questions about the agents, answered from data.

It is the first consumer of the platform's MCP client: its only tools are the
analytics MCP server's (`agent_platform.knowledge.mcp.server.analytics`), arriving as
ordinary governed tools, authorized by the Gatekeeper, run by the Executor and
audited like any other.

Importing this package declares its configuration layer, as every agent does:
the dependency points from the agent to the platform. `AGENTIC_OVERLAYS` still
wins when a deployment, the Makefile or a test has already decided.
"""

import os
from pathlib import Path

CONFIG_FILE = Path(__file__).resolve().parent / "config" / "config.yaml"
_OVERLAYS_ENV = "AGENTIC_OVERLAYS"


def _declare_config_layer() -> None:
    """Name this agent's overlay, unless the environment already did."""
    if not os.environ.get(_OVERLAYS_ENV):
        os.environ[_OVERLAYS_ENV] = str(CONFIG_FILE)


_declare_config_layer()
