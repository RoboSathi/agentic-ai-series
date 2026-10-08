# ADR 0008 — MCP unparked: the analytics server, the client, and the ops agent

- Status: accepted
- Date: 2026-10-08
- Supersedes: LLD §16.1 "parked, interface-only"

## Context

`agent_platform/mcp` held the MCP contracts only. Natural-language questions
over the analytics mart need an MCP server, and a demo of it needs a client the
series can stand behind — one that takes a remote tool through the same
governance as a local one.

## Decision

- **Server** (`agent_platform.analytics.mcp_server`, MCP SDK v2 `MCPServer`,
  stdio): three read-only tools — `list_views`, `describe_view`, `run_query` —
  and a `mart://manifest` resource. Two walls: `QueryGuard` parses the SQL with
  sqlglot and allows one SELECT over curated views or its own CTEs, no table
  functions, no denied functions, wrapped in an outer LIMIT; and the database is
  in-memory, loaded from the Parquet, with external access disabled and its
  configuration locked, under a deadline (`connection.interrupt`) and a byte cap.
  A refusal is a tool error whose text tells the model what to change.
- **Client** (`agent_platform.mcp`): `SdkMcpTransport` (stdio/HTTP from an
  agent's `mcp.servers`), `SdkMcpSession` (one session owned by its own task —
  the SDK's cancel scopes belong to the task that entered them, and tool calls
  arrive from whichever task runs the turn), `GovernedMcpToolSource` (tools as
  `ToolDefinition`s; arguments validated by a model built from the tool's
  schema; envelope from `ConfiguredGovernancePolicy`, defaulting to the most
  restrictive `mcp.default_envelope`). The server's own annotations are not
  trusted.
- **Ops agent** (`src/ops_agent`, `make agent ops`): the first consumer. A
  LangGraph ReAct loop whose only tools are the analytics server's; each call
  is validated, authorized by the Gatekeeper against `ops_agent.policy`, run by
  the Executor and audited. Its `hallucination_guard` is a deterministic
  figures check: every number in the answer must appear in a query result. The
  reply shows the SQL and its rows.

## Consequences

- Any MCP client (Claude Code, MCP Inspector) can query the mart:
  `make analytics-mcp`.
- `RegistryToolHandler` lets the Executor run self-implementing tools.
- Owed: an NL-to-SQL golden set scored by execution accuracy in the platform
  eval framework (not built as a parallel runner); HTTP transport auth.
