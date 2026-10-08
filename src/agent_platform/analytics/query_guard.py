"""Which SQL a model may run over the mart: one SELECT, over the curated views only.

A model writes the SQL, so the SQL is untrusted input. It is parsed (never
pattern-matched) and refused unless it is a single query statement whose every
relation is a curated view or one of its own CTEs, with no table functions and
no denied functions. The query is then wrapped in an outer LIMIT, so no answer
is larger than `analytics.query.max_rows` whatever the model asked for.

This is one of two walls. The database the server queries is in memory, loaded
from the mart, with external access disabled and its configuration locked, so a
query the parser misjudged still cannot reach a file, a setting or the network.
"""

from collections.abc import Iterable

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from agent_platform.config import get_settings
from agent_platform.config.settings import AnalyticsQuerySettings


class QueryRejectedError(ValueError):
    """The SQL is not one a model may run; the message says why, for the model to fix."""


class QueryGuard:
    """Checks one model-written query and bounds it (Strategy)."""

    def __init__(
        self, views: Iterable[str], settings: AnalyticsQuerySettings | None = None
    ) -> None:
        self._views = {view.lower() for view in views}
        self._settings = settings or get_settings().analytics.query
        self._denied = {name.lower() for name in self._settings.denied_functions}

    def check(self, sql: str) -> str:
        """The query, bounded, or `QueryRejectedError` saying what to change."""
        statement = self._single_statement(sql)
        self._check_relations(statement)
        self._check_functions(statement)
        bounded = f"SELECT * FROM ({statement.sql(dialect=self._settings.dialect)}) AS q"
        return f"{bounded} LIMIT {self._settings.max_rows + 1}"

    def _single_statement(self, sql: str) -> exp.Query:
        words = self._settings.rejections
        try:
            statements = [s for s in sqlglot.parse(sql, read=self._settings.dialect) if s]
        except ParseError as exc:
            raise QueryRejectedError(words.unparseable.format(error=exc)) from None
        if len(statements) != 1:
            raise QueryRejectedError(words.single_statement)
        statement = statements[0]
        if not isinstance(statement, exp.Query):
            raise QueryRejectedError(words.not_a_select)
        return statement

    def _check_relations(self, statement: exp.Query) -> None:
        words = self._settings.rejections
        ctes = {cte.alias_or_name.lower() for cte in statement.find_all(exp.CTE)}
        for table in statement.find_all(exp.Table):
            if not isinstance(table.this, exp.Identifier):
                raise QueryRejectedError(words.table_function.format(name=table.sql()))
            if table.args.get("db") or table.args.get("catalog"):
                raise QueryRejectedError(words.qualified_name.format(name=table.sql()))
            if table.name.lower() not in self._views | ctes:
                raise QueryRejectedError(words.unknown_view.format(name=table.name))

    def _check_functions(self, statement: exp.Query) -> None:
        for function in statement.find_all(exp.Func):
            name = function.name if isinstance(function, exp.Anonymous) else function.sql_name()
            if name.lower() in self._denied:
                raise QueryRejectedError(
                    self._settings.rejections.denied_function.format(name=name)
                )


__all__ = ["QueryGuard", "QueryRejectedError"]
