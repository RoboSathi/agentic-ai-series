"""`/stats` — this session at a glance: turns, model calls, tools, decisions, SLOs.

Session-scoped and live: it reads the fact tables and audit trail of the agent
running in this console (the same in-process database, so no lock to share),
narrowed to the console's session. The fleet-wide, historical view is the
dashboard (`make dashboard`); this is the one an operator glances at mid-demo.
SLOs are computed here over the session's turns with the same rule the mart and
the registry use (`LOWER_IS_BETTER`).
"""

from typing import Any

from agent_platform.analytics.catalog import load_catalog, render_sql
from agent_platform.commands.scope import ConsoleScope
from agent_platform.commands.types import CommandResult
from agent_platform.config import get_settings
from agent_platform.observability.facts import (
    LlmCallFact,
    ToolCallFact,
    TurnOutcomeFact,
    fact_table_ddl,
)
from agent_platform.observability.slo import LOWER_IS_BETTER
from agent_platform.observability.store import DuckDbObservabilityStore
from agent_platform.presentation.text_tree import TreeNode, tree_lines

_P95 = "coalesce(quantile_cont({column}, 0.95), 0)"
_SLO_VIEW = "slo_observations"


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
        root.add(self._words.turns.format(**turns))
        self._models(
            root.add(self._words.models.format(**self._one(_CALLS, agent, session))), agent, session
        )
        self._tools(
            root.add(self._words.tools.format(**self._one(_TOOLS, agent, session))), agent, session
        )
        root.add(self._words.decisions.format(counts=self._decisions(agent, session)))
        self._slos(root.add(self._words.slos), agent, session)
        result = CommandResult(title=root.label)
        for line in tree_lines(root)[1:]:
            result.line(line)
        return result

    def _models(self, node: TreeNode, agent: str, session: str) -> None:
        for row in self._all(_CALLS_BY_MODEL, agent, session):
            node.add(self._words.model_line.format(**row))

    def _tools(self, node: TreeNode, agent: str, session: str) -> None:
        for row in self._all(_TOOLS_BY_ACTION, agent, session):
            node.add(self._words.tool_line.format(**row))

    def _decisions(self, agent: str, session: str) -> str:
        rows = self._all(_DECISIONS, agent, session)
        return " · ".join(f"{row['label']} {row['n']}" for row in rows) or self._words.none

    def _slos(self, node: TreeNode, agent: str, session: str) -> None:
        targets = get_settings().observability.slo_targets
        view = load_catalog().get(_SLO_VIEW)
        if view is None:
            return
        observations = render_sql(view, lambda name: name)
        sql = (
            f"SELECT metric, avg(value) AS observed, count(*) AS samples FROM ({observations}) "
            "WHERE agent_id = ? AND session_id = ? GROUP BY 1 ORDER BY 1"
        )
        for row in self._all(sql, agent, session):
            target = targets.get(row["metric"])
            if target is None:
                continue
            lower = row["metric"] in LOWER_IS_BETTER
            met = row["observed"] <= target if lower else row["observed"] >= target
            verdict = self._words.met if met else self._words.missed
            node.add(self._words.slo_line.format(target=target, verdict=verdict, **row))

    def _one(self, sql: str, agent: str, session: str) -> dict[str, Any]:
        rows = self._all(sql, agent, session)
        return rows[0] if rows else {}

    def _all(self, sql: str, agent: str, session: str) -> list[dict[str, Any]]:
        return list(self._db().iter_rows(sql, (agent, session)))

    def _db(self) -> DuckDbObservabilityStore:
        if self._store is None:
            self._store = DuckDbObservabilityStore()
            for fact in (LlmCallFact, ToolCallFact, TurnOutcomeFact):
                self._store.execute_script(fact_table_ddl(fact))
        return self._store


_SESSION = "WHERE agent_id = ? AND session_id = ?"
_TURNS = (
    "SELECT count(*) AS turns, count(*) FILTER (WHERE answered) AS answered, "
    f"count(*) FILTER (WHERE escalated) AS escalated, {_P95.format(column='latency_ms')} AS p95_ms "
    f"FROM platform_turn_outcomes {_SESSION}"
)
_CALLS = (
    "SELECT count(*) AS calls, coalesce(sum(total_tokens), 0) AS tokens, "
    "count(*) FILTER (WHERE position > 0) AS backups, count(*) FILTER (WHERE NOT succeeded) AS failed, "
    f"{_P95.format(column='latency_ms')} AS p95_ms FROM platform_llm_calls {_SESSION}"
)
_CALLS_BY_MODEL = (
    "SELECT role, model, count(*) AS calls, sum(total_tokens) AS tokens, "
    f"{_P95.format(column='latency_ms')} AS p95_ms FROM platform_llm_calls {_SESSION} "
    "GROUP BY 1, 2 ORDER BY calls DESC"
)
_TOOLS = (
    "SELECT count(*) AS calls, count(*) FILTER (WHERE NOT succeeded) AS failed, "
    f"{_P95.format(column='duration_ms')} AS p95_ms FROM platform_tool_calls {_SESSION}"
)
_TOOLS_BY_ACTION = (
    "SELECT action, count(*) AS calls, count(*) FILTER (WHERE NOT succeeded) AS failed "
    f"FROM platform_tool_calls {_SESSION} GROUP BY 1 ORDER BY calls DESC"
)
_DECISIONS = (
    "SELECT decision || coalesce('/' || json_extract_string(authority_context, '$.standing'), '') "
    "AS label, count(*) AS n FROM platform_audit_entries "
    "WHERE agent_id = ? AND json_extract_string(authority_context, '$.session_id') = ? "
    "GROUP BY 1 ORDER BY n DESC"
)


__all__ = ["StatsCommand"]
