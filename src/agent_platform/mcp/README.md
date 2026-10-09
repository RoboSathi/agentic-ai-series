# mcp — Model Context Protocol client

Tools that arrive over MCP rather than being written as Python functions.
Whatever an MCP server offers enters as an ordinary `ToolDefinition` through
the same registry, so it takes the platform's single tool path
(`tools/path.py`): validated, duplicate-checked, ruled on by the Gatekeeper
(or run UNGOVERNED, audited), executed by the Executor, recorded as a
`fact_tool_call`. There is no side door.

## Vocabulary

- **McpToolSource** — a tool source backed by an MCP session (`GovernedMcpToolSource`)
- **ToolGovernancePolicy** — the envelope this deployment grants a server's tools,
  whatever the server says about itself (`ConfiguredGovernancePolicy`)
- **McpSession** — one live client session (`SdkMcpSession`, an Active Object owning
  the SDK client in its own task)
- **McpTransport** — how a session is opened (`SdkMcpTransport`: stdio, or an
  injected resolver for an in-process server)

## Public interface

- `McpServerSpec`, `McpSession`, `McpToolInfo`, `McpCallResult`, `McpToolSource`,
  `McpTransport`, `ToolGovernancePolicy`
- `SdkMcpSession`, `SdkMcpTransport`, `server_spec`, `GovernedMcpToolSource`,
  `ConfiguredGovernancePolicy`

## Configuration

`mcp.servers[]` in an agent's overlay: name, transport, command (`{python}` is
this interpreter), namespace and envelope (surface, side effects, required
permission).

## Seams

- Plugs into `tools.NamespacedToolRegistry` via `McpToolSource.tools()`.
- The platform's own server is `agent_platform.analytics.mcp_server`
  (list_metrics, compute_metric, list_views, describe_view, run_query).

## Collaborators

- `tools`, `gatekeeper`, `executor`, `analytics`.

## Tests

`tests/agent_platform/test_mcp_client.py` (in-process server; owed a rewrite for
the ADR 0009 analytics source).

## Status and limits

Implemented (ADR 0008). Used by the ops agent against the analytics server.
