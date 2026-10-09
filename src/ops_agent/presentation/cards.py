"""What a tool returned, as its TOOL RESULT card shows it: a summary line and a preview.

Each analytics tool answers with structured content of its own shape: a query (or
a computed metric) returns SQL, columns and rows; `list_metrics` the catalogue;
`list_views` the views; `describe_view` one view's columns. The card names what
came back in those terms, never "0 rows" for something that has no rows.
"""

from typing import Any

from ops_agent.config.settings import OpsAgentSettings

_QUERY_KEYS = frozenset({"sql", "columns", "rows"})
_METRICS, _VIEWS, _COLUMNS, _NAME = "metrics", "views", "columns", "name"


def result_card(
    tool: str, data: dict[str, Any], settings: OpsAgentSettings
) -> tuple[str, dict[str, Any]]:
    """(summary, preview fields) for one successful tool result."""
    cards = settings.cards
    if _QUERY_KEYS <= data.keys():
        rows = list(data.get("rows") or [])
        preview = {
            "sql": data.get("sql", ""),
            "columns": data.get("columns", []),
            "rows": rows[: settings.table_rows],
            "row_count": data.get("row_count", len(rows)),
            "truncated": data.get("truncated", False),
        }
        return cards.result_rows.format(tool=tool, rows=preview["row_count"]), preview
    if _METRICS in data:
        names = [str(item.get(_NAME, "")) for item in data[_METRICS]]
        return cards.result_metrics.format(tool=tool, count=len(names)), {_METRICS: names}
    if _VIEWS in data:
        names = [str(item.get(_NAME, "")) for item in data[_VIEWS]]
        return cards.result_views.format(tool=tool, count=len(names)), {_VIEWS: names}
    if _COLUMNS in data and _NAME in data:
        columns = [f"{column.get(_NAME)}: {column.get('type', '')}" for column in data[_COLUMNS]]
        summary = cards.result_columns.format(tool=tool, count=len(columns), view=data[_NAME])
        return summary, {_COLUMNS: columns}
    return cards.result_done.format(tool=tool), {}


__all__ = ["result_card"]
