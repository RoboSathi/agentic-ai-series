# Ops agent — patterns

Where this agent uses each design pattern, derived from the code as it stands.

## Workflow and agentic patterns

| Pattern | Where | Notes |
|---|---|---|
| react | graph/graph.py · reason ⇄ act | the model explores the mart (list → describe → query) and corrects its SQL from the server's refusals, bounded by `ops_agent.max_steps` |
| tool_use (MCP) | graph/builder.py · `_sources` → `GovernedMcpToolSource` | the only tools are a remote MCP server's, arriving as ordinary governed `ToolDefinition`s |
| guardrails | graph/agent.py · `_screened` | the person's message is screened for injection before the model reads it (`prompt_injection`) |
| hallucination_protection | guardrails/figures.py · `FigureCheck` | every figure in the answer must appear in a query result; the rest are reported as unverified (`hallucination_guard`) |
| human-verifiable output | presentation/reply.py | the reply carries the SQL it ran and the rows it rests on |

## Governance and LLD patterns

| Pattern | Where |
|---|---|
| gatekeeper (authority only) | graph/tool_calls.py · `GovernedCalls` — VALID when allowed, VOID refused, UNGOVERNED when withheld |
| adapter | agent_platform/mcp/session.py (SDK client), executor/registry_handler.py |
| active object | agent_platform/mcp/session.py · one task owns the MCP session |
| strategy | agent_platform/analytics/query_guard.py |
| facade | agent_platform/analytics/mart_db.py · `MartDatabase` |
| factory / composition root | graph/builder.py — features are read once, here |
