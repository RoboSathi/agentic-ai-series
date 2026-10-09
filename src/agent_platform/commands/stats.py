"""`/stats` — this session at a glance: tasks, turns, model calls, tools, checks, decisions, SLOs.

Session-scoped and live: it reads the fact tables and audit trail of the agent
running in this console (the same in-process database, so no lock to share),
narrowed to the console's session. The fleet-wide, historical view is the
dashboard (`make dashboard`); this is the one an operator glances at mid-demo.
The SLO lines are the metric catalogue's own formulas (`analytics/semantic_layer.yaml`)
over the session's rows, so they cannot disagree with the source.
"""

from typing import Any

from agent_platform.analytics.catalog import Catalog, MetricDefinition, load_catalog, render_sql
from agent_platform.commands.scope import ConsoleScope
from agent_platform.commands.types import CommandResult
from agent_platform.config import get_settings
from agent_platform.observability.facts import TIER1_FACTS, fact_table_ddl
from agent_platform.observability.store import DuckDbObservabilityStore
from agent_platform.presentation.text_tree import TreeNode, tree_lines

_P95 = "coalesce(quantile_cont({column}, 0.95), 0)"


class StatsCommand:
    """This console session's numbers, as a tree."""

    group = "Inspect"
    name = "stats"

    def __init__(self, store: DuckDbObservabilityStore | None = None) -> None:
        self._store = store
        self._scope: ConsoleScope | None = None
        self._words = get_settings().commands.stats
        self.summary = self._words.summary
        self.usage = self._words.usage

    def bind_scope(self, scope: ConsoleScope) -> None:
        self._scope = scope

    async def run(self, args: list[str]) -> CommandResult:
        session = self._scope.session_id if self._scope is not None else ""
        agent = get_settings().agent.id
        root = TreeNode(self._words.title.format(session=session or self._words.none, agent=agent))
        turns = self._one(_TURNS, agent, session)
        if not turns or not turns["turns"]:
            return CommandResult(title=root.label).line(self._words.no_session)
        self._lines(root.add(self._words.tasks.format(**self._one(_TASKS, agent, session))),
                    _TASKS_BY_OUTCOME, self._words.task_line, agent, session)  # fmt: skip
        root.add(self._words.turns.format(**turns))
        self._lines(root.add(self._words.models.format(**self._one(_CALLS, agent, session))),
                    _CALLS_BY_MODEL, self._words.model_line, agent, session)  # fmt: skip
        self._lines(root.add(self._words.tools.format(**self._one(_TOOLS, agent, session))),
                    _TOOLS_BY_NAME, self._words.tool_line, agent, session)  # fmt: skip
        self._lines(root.add(self._words.checks.format(**self._one(_CHECKS, agent, session))),
                    _CHECKS_BY_NAME, self._words.check_line, agent, session)  # fmt: skip
        root.add(self._words.decisions.format(counts=self._decisions(agent, session)))
        self._slos(root.add(self._words.slos), agent, session)
        result = CommandResult(title=root.label)
        for line in tree_lines(root)[1:]:
            result.line(line)
        return result

    def _lines(self, node: TreeNode, sql: str, wording: str, agent: str, session: str) -> None:
        for row in self._all(sql, agent, session):
            node.add(wording.format(**row))

    def _decisions(self, agent: str, session: str) -> str:
        rows = self._all(_DECISIONS, agent, session)
        return " · ".join(f"{row['label']} {row['n']}" for row in rows) or self._words.none

    def _slos(self, node: TreeNode, agent: str, session: str) -> None:
        """Each catalogue metric with an SLO target, by its own formula, over this session."""
        targets = get_settings().observability.slo_targets
        catalog = load_catalog()
        for metric in catalog.metrics:
            target = targets.get(metric.name)
            observed = self._metric(catalog, metric, agent, session) if target is not None else None
            if target is None or observed is None:
                continue
            met = observed <= target if metric.lower_is_better else observed >= target
            verdict = self._words.met if met else self._words.missed
            node.add(
                self._words.slo_line.format(
                    metric=metric.name, observed=observed, target=target, verdict=verdict
                )
            )

    def _metric(
        self, catalog: Catalog, metric: MetricDefinition, agent: str, session: str
    ) -> float | None:
        view = catalog.get(metric.view)
        if view is None:
            return None
        source = render_sql(view, lambda name: name, catalog)
        sql = f"SELECT CAST({metric.expr} AS DOUBLE) AS value FROM ({source}) {_SESSION}"
        row = self._one(sql, agent, session)
        value = row.get("value")
        return float(value) if value is not None else None

    def _one(self, sql: str, agent: str, session: str) -> dict[str, Any]:
        rows = self._all(sql, agent, session)
        return rows[0] if rows else {}

    def _all(self, sql: str, agent: str, session: str) -> list[dict[str, Any]]:
        return list(self._db().iter_rows(sql, (agent, session)))

    def _db(self) -> DuckDbObservabilityStore:
        if self._store is None:
            self._store = DuckDbObservabilityStore()
            for fact in TIER1_FACTS:
                self._store.execute_script(fact_table_ddl(fact))
        return self._store


_SESSION = "WHERE agent_id = ? AND session_id = ?"
_CLOSED = "event = 'closed'"
_TASKS = (
    "SELECT count(DISTINCT task_id) AS tasks, "
    f"count(DISTINCT task_id) FILTER (WHERE {_CLOSED} AND success AND terminal) AS resolved, "
    "count(DISTINCT task_id) - count(DISTINCT task_id) FILTER (WHERE "
    f"{_CLOSED}) AS open FROM fact_task_event {_SESSION}"
)
_TASKS_BY_OUTCOME = (
    "SELECT kind, outcome, count(*) AS n FROM fact_task_event "
    f"{_SESSION} AND {_CLOSED} GROUP BY 1, 2 ORDER BY n DESC"
)
_TURNS = (
    "SELECT count(*) AS turns, count(*) FILTER (WHERE status = 'crashed') AS crashed, "
    "count(*) FILTER (WHERE status = 'cancelled') AS cancelled, coalesce(sum(steps), 0) AS steps, "
    f"{_P95.format(column='latency_ms')} AS p95_ms FROM fact_turn {_SESSION}"
)
_CALLS = (
    "SELECT count(*) AS calls, coalesce(sum(total_tokens), 0) AS tokens, "
    "count(*) FILTER (WHERE is_backup) AS backups, count(*) FILTER (WHERE NOT succeeded) AS failed, "
    f"{_P95.format(column='latency_ms')} AS p95_ms FROM fact_model_call {_SESSION}"
)
_CALLS_BY_MODEL = (
    "SELECT role, model, count(*) AS calls, sum(total_tokens) AS tokens, "
    f"{_P95.format(column='latency_ms')} AS p95_ms FROM fact_model_call {_SESSION} "
    "GROUP BY 1, 2 ORDER BY calls DESC"
)
_TOOLS = (
    "SELECT count(*) AS calls, count(*) FILTER (WHERE stage = 'executed') AS executed, "
    "count(*) FILTER (WHERE stage = 'executed' AND NOT succeeded) AS failed, "
    f"{_P95.format(column='duration_ms')} AS p95_ms FROM fact_tool_call {_SESSION}"
)
_TOOLS_BY_NAME = (
    "SELECT tool, count(*) AS calls, "
    "string_agg(stage || ' ' || n, ', ' ORDER BY n DESC) AS stages FROM ("
    f"SELECT tool, stage, count(*) AS n FROM fact_tool_call {_SESSION} GROUP BY 1, 2) "
    "GROUP BY tool ORDER BY calls DESC"
)
_CHECKS = (
    "SELECT count(*) AS checks, count(*) FILTER (WHERE verdict = 'block') AS blocked, "
    "count(*) FILTER (WHERE verdict = 'flag') AS flagged, "
    "count(*) FILTER (WHERE verdict IN ('error', 'fail_open')) AS failed "
    f"FROM fact_check {_SESSION}"
)
_CHECKS_BY_NAME = (
    "SELECT check_name, purpose, sum(n) AS n, "
    "string_agg(verdict || ' ' || n, ', ' ORDER BY n DESC) AS verdicts FROM ("
    f"SELECT check_name, purpose, verdict, count(*) AS n FROM fact_check {_SESSION} "
    "GROUP BY 1, 2, 3) GROUP BY 1, 2 ORDER BY n DESC"
)
_DECISIONS = (
    "SELECT decision || coalesce('/' || json_extract_string(authority_context, '$.standing'), '') "
    "AS label, count(*) AS n FROM platform_audit_entries "
    "WHERE agent_id = ? AND json_extract_string(authority_context, '$.session_id') = ? "
    "GROUP BY 1 ORDER BY n DESC"
)


__all__ = ["StatsCommand"]
