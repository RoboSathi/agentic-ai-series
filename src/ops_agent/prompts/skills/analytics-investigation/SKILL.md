---
name: analytics-investigation
description: Answer a question about how the agents are running from the analytics source — a catalogue metric when one is named, otherwise one SQL query over the described views.
triggers: [always]
keywords: [metric, query, sql, views, latency, tokens, availability, slo]
prefetch: [list_metrics, list_views]
tools: [list_metrics, compute_metric, list_views, describe_view, run_query]
---

## Preconditions
- The question can be answered from the analytics source.

## Procedure
1. The metric catalogue and the view list are already above.
2. For a named metric, call `compute_metric` with its name (and agent, days or breakdown as
   the question needs): the catalogue's formula is the definition.
3. Otherwise call `describe_view` for each view you will use, then `run_query` with ONE
   DuckDB SELECT over them. Aggregate in SQL rather than fetching rows.

## Boundaries
- Never state a figure no tool returned.

## Completion
Stop once the question is answered, or once it is clear the data cannot answer it.
