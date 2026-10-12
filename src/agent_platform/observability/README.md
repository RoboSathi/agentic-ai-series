# observability — cards, facts, traces, SLOs

Everything the harness reports about itself: the live status feed and cards, Tier-1 facts at platform choke points (turn bracket, task API, check recorder), the observability store, LangSmith export, cost and SLOs.

- **Entry points:** `StageReporter`, `StatusEvent`, `FactStore`, `units.py`, `checks.py`, `DuckDbObservabilityStore`, `LangSmithExporter`, `SLORegistry`
- **Configuration:** `observability:`
- **Design:** [LLD §A20](../../../design_docs/LLD.md#a20-observability) · [ADR P-24](../../../design_docs/ADR.md#p-24--analytics-is-a-platform-contract)
