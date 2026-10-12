"""The analytics source: curated views over the facts, the audit trail and the evals.

`catalog` is the semantic layer (each view's SQL and column meanings, the
metric catalogue, and every agent's registered layer), `metric_sql` renders
catalogue metrics,
`exporter` writes the source as Parquet, `query_guard` decides which SQL a
model may run over it. The MCP server over it is `agent_platform.knowledge.mcp.server.analytics`.
Readers use the source, never the live databases.
"""

from agent_platform.analytics.catalog import (
    Catalog,
    CuratedView,
    MetricDefinition,
    load_catalog,
    render_sql,
)
from agent_platform.analytics.exporter import (
    ExportedView,
    ParquetExporter,
    SourceManifest,
    ViewContractError,
    read_manifest,
)

__all__ = [
    "Catalog",
    "CuratedView",
    "ExportedView",
    "ViewContractError",
    "ParquetExporter",
    "SourceManifest",
    "MetricDefinition",
    "load_catalog",
    "read_manifest",
    "render_sql",
]
