"""The analytics mart: curated views over the facts, the audit trail and the evals.

`catalog` is the semantic layer (each view's SQL and column meanings),
`exporter` writes the mart as Parquet, `query_guard` decides which SQL a
model may run over it, and `mcp_server` serves it over the Model Context
Protocol. Readers use the mart, never the live databases.
"""

from agent_platform.analytics.catalog import Catalog, CuratedView, load_catalog, render_sql
from agent_platform.analytics.exporter import (
    ExportedView,
    MartContractError,
    MartExporter,
    MartManifest,
    read_manifest,
)

__all__ = [
    "Catalog",
    "CuratedView",
    "ExportedView",
    "MartContractError",
    "MartExporter",
    "MartManifest",
    "load_catalog",
    "read_manifest",
    "render_sql",
]
