# Dashboards

Two Rill dashboards over the analytics mart (`data/analytics/*.parquet`).

```
make dashboard-install   # Rill into .tools/rill, checksum-verified, repo-local (skips if current)
make dashboard           # links the mart as dashboards/rill/data and starts Rill (fresh load)
make dashboard-refresh   # re-export the mart and reload a running Rill (it does not watch Parquet)
```

| Dashboard | Grain (mart view) | The question it answers |
|---|---|---|
| **Agent Health** | one row per turn (`turns`) | Are the agents doing their job, safely, fast enough, and at what cost? Answered / escalated / loop-aborted rate (the SLOs), p50/p95 turn latency, tokens per turn, model calls per turn, backup share, model and tool failure rates, actions denied / escalated / ungoverned, unverified claims — by agent, outcome, primary model, channel. |
| **Calls & Decisions** | one row per event (`events`) | Which model, tool or rule is the problem? Every model call, tool call and authority decision: failure rate, p95 latency, tokens, backup rate, deny and escalate rate, ungoverned count — by kind, name, outcome, standing, role. Holds the audit history from before turn facts existed. |

Agent Health is empty until an agent has run a turn (facts are recorded per turn).
Evals are not on a dashboard; ask the ops agent (`make agent ops`) about `eval_runs`.

Column meanings live in `src/agent_platform/analytics/semantic_layer.yaml`;
`tests/agent_platform/test_rill_contract.py` fails if a dashboard names a column
the mart does not export.
