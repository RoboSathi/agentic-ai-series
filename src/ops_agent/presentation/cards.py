"""What a tool returned, as its TOOL RESULT card shows it: a summary line and a preview.

Each analytics tool answers with structured content of its own shape: a query (or
a computed metric) returns SQL, columns and rows; `list_metrics` the catalogue;
`list_views` the views; `describe_view` one view's columns. The card names what
came back in those terms, never "0 rows" for something that has no rows.
"""

from typing import Any

from agent_platform.observability.step_card import Line, StepCard
from agent_platform.observability.types import StatusKind
from ops_agent.config.settings import OpsAgentSettings, get_agent_settings
from ops_agent.domain import QueryRecord
from ops_agent.guardrails.sql_review import SqlReview

# The SQL review's card is the platform's REFLECT step card.
REFLECT_STAGE = "reflect"
_PASS, _WARN = "pass", "warn"

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


def review_card(
    review: SqlReview, queries: list[QueryRecord], question: str, used: int, again: bool
) -> tuple[StatusKind, str, StepCard]:
    """(status, heading, card) for one SQL review: asked, the model's part, the decision."""
    settings = get_agent_settings().sql_review
    words = settings.card
    card = StepCard().asked(
        words.asked.format(question=question),
        *(
            words.query_line.format(number=number, sql=" ".join(query.sql.split()))
            for number, query in enumerate(queries, 1)
        ),
    )
    card.model(review.call, response=False)
    if review.failed:
        decision = words.decision_failed.format(error=review.failed)
        return "warning", words.failed_heading, card.decision(_toned(decision, _WARN))
    card.reasoning(*([review.rationale] if review.rationale else []))
    card.response(*review.issues, *([review.hint] if review.hint else []))
    issues = words.issues_suffix.format(count=len(review.issues)) if review.issues else ""
    heading = words.heading.format(verdict=review.verdict.upper(), issues=issues)
    if not review.revise:
        return "completed", heading, card.decision(_toned(words.decision_accept, _PASS))
    limit = settings.max_reflections
    decided = (
        words.decision_revise.format(used=used, max=limit)
        if again
        else words.decision_capped.format(max=limit)
    )
    return "warning", heading, card.decision(_toned(decided, _WARN))


def _toned(text: str, tone: str) -> Line:
    """A card line in a tone (`tui.tone_styles`): the decision reads in its outcome's colour."""
    return {"text": text, "tone": tone}


__all__ = ["REFLECT_STAGE", "result_card", "review_card"]
