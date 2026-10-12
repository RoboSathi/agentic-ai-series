# analytics/dashboards — generated Rill dashboards

`make dashboard-install` installs Rill here (`.rill/`, gitignored); `make dashboard` re-exports the source, generates the project under `rill/` from the semantic layer and serves it; `make dashboard-refresh` reloads a running one. Only `rill.yaml` and the scripts are versioned.

- **Entry points:** `install_rill.sh`, `serve_rill.sh`, `rill/rill.yaml`
- **Configuration:** `analytics.dashboards`
- **Design:** [LLD §A26](../../../../design_docs/LLD.md#a26-analytics) · [ADR P-26](../../../../design_docs/ADR.md#p-26--dashboards-are-generated-not-hand-written)
