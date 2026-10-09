# Ops agent

Answers questions about how the platform's agents are running — latency,
tokens, backups, tool failures, Gatekeeper decisions, SLOs, eval quality — by
querying the analytics source through the platform's analytics MCP server.

```
make analytics-export      # or keep any agent's console running: it refreshes the source
make agent ops             # ask: "p95 model latency by role this week?"
```

Its harness (`features:` in `config/config.yaml`, moved with `/harness`):
`prompt_injection`, `gatekeeper`, `hallucination_guard`. The MCP server's
tools run under the envelope in `mcp.servers[].envelope`, not under anything
the server claims. See ADR 0007 (the source) and ADR 0008 (MCP).
