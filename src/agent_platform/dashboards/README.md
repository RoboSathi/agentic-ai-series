# Dashboards

Rill dashboards over the active analytics source's Parquet (`analytics.sources`, `SOURCE=`), **generated**
from the semantic layer by `python -m agent_platform.analytics dashboards` (ADR 0009 D3).
Nothing under `rill/models`, `rill/metrics` or `rill/dashboards` is hand-written or
versioned, nor is Rill's runtime state (`rill/tmp`), the `rill/data` link or the binary (`.rill/`): a measure is the metric catalogue's own formula, so it cannot drift.

```
make dashboard-install   # Rill into .rill/ here (gitignored), checksum-verified, repo-local (skips if current)
make dashboard           # re-exports the source, links it as rill/data, generates the project, starts Rill
                         # (RILL_PORT / RILL_GRPC_PORT to run a second Rill beside one already running)
make dashboard-refresh   # re-export the source, regenerate, and reload a running Rill
```

Four dashboards (Rill *canvases*), one page each:

| Dashboard | Shows |
|---|---|
| **Agent Platform** | Every agent together: each metric over all their traffic, and in every section the agents ranked side by side. |
| **Each agent** | Its own traffic: the platform's sections, then any the agent declares (its own views in the semantic layer). |

Every dashboard has the same sections, in this order — **Outcomes, Hallucination &
grounding, Security, Tool use, Reliability & speed, Cost, Models** — each a row of KPI tiles
(with change against the previous period), trend lines, and rankings or tables:

- **Security** has one block per screen the agent runs: *PII* (found), *Prompt injection*
  (blocked, guard failures), *Content moderation* (blocked, flagged, guard failures), then
  screen coverage and a table by screen. A screen the agent doesn't run shows no tiles.
- **Models** has a row per model the agent used — each LLM (by role), DeBERTa NLI, the
  jailbreak guard, content safety, embeddings, reranker, PII NER, intent — with calls,
  latency p50/p95/p99, failure rate and tokens per call.
- On the **Agent Platform** page each section ends with a table of the agents side by side.
Every measure is named "label (what it means)", e.g. *Resolution rate (succeeded ÷ finished
tasks)*; hover for the full definition. Counts are named for what they count (*Tasks*,
*Turns*, *Tool proposals*), never rows. Each measure is the metric catalogue's own formula over
the raw rows, so a number is exact for whatever range you pick, and matches MCP `compute_metric`.
Eval traffic is excluded.

To see every page filled without running agents, generate a demo world (30 days, all
three agents, one incident day) and point the dashboard at it:

```
make dummy-data analytics
make dashboard SOURCE=dummy
```

An agent appears on the Agent Platform page, and gets its own dashboard, by running turns: no analytics code. Its own
sections come from `src/<package>/analytics/semantic_layer.yaml` (`groups:`), if it registers one.

Evals are not on a dashboard yet (the evals rework, D8); ask the ops agent
(`make agent ops`) about `eval_runs`.
