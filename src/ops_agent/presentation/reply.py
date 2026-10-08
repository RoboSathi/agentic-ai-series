"""The reply: the model's answer, then the SQL it ran and the rows that came back.

An answer about data is only as good as the query behind it, so the query is
shown, not hidden: the operator can read it, copy it, and run it themselves.
Plain lines, like every platform command's output, so the same text reads in
the TUI and in a log.
"""

from typing import Any

from agent_platform.presentation.text_table import Align, box_table
from ops_agent.config.settings import OpsAgentSettings
from ops_agent.domain import QueryRecord


def present(
    answer: str, queries: list[QueryRecord], unsupported: list[str], settings: OpsAgentSettings
) -> str:
    """The reply as the operator reads it."""
    words = settings.words
    parts = [answer.strip()]
    if unsupported:
        parts.append(words.ungrounded.format(numbers=", ".join(unsupported)))
    if queries:
        parts.append("\n".join(_query_lines(queries[-1], settings)))
        parts.append(words.sources_heading.format(count=len(queries)))
    return "\n\n".join(part for part in parts if part)


def _query_lines(query: QueryRecord, settings: OpsAgentSettings) -> list[str]:
    words = settings.words
    shown = query.rows[: settings.table_rows]
    more = query.truncated or len(query.rows) > len(shown)
    lines = [words.query_heading, *(f"  {line}" for line in query.sql.strip().splitlines())]
    lines.append(
        words.rows_heading.format(
            shown=len(shown), total=query.row_count, more=words.more_rows if more else ""
        )
    )
    if query.columns and shown:
        cells = [[_cell(value, settings.cell_chars) for value in row] for row in shown]
        align: list[Align] = [
            "right" if _numeric(shown, i) else "left" for i in range(len(query.columns))
        ]
        lines.extend(box_table(query.columns, cells, align))
    return lines


def _cell(value: Any, limit: int) -> str:
    text = "" if value is None else str(round(value, 4)) if isinstance(value, float) else str(value)
    return text if len(text) <= limit else f"{text[: limit - 1]}…"


def _numeric(rows: list[list[Any]], column: int) -> bool:
    return all(isinstance(row[column], int | float) for row in rows if row[column] is not None)


__all__ = ["present"]
