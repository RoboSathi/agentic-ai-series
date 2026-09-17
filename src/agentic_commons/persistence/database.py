"""SQLite-backed persistence."""

import sqlite3
from pathlib import Path
from typing import Any

from agentic_commons.logging import log_call
from agentic_commons.persistence.types import ExecutionResult

_DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "agentic.db"


class Database:
    """SQLite database interface.

    The db path is resolved relative to the package by default, never the
    shell's working directory, so it works the same regardless of where
    the process is invoked from.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or _DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

    @log_call("persistence")
    def get_connection(self) -> sqlite3.Connection:
        """Open a new connection with row access by column name."""
        connection = sqlite3.connect(self._db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @log_call("persistence")
    def execute(
        self, query: str, params: tuple[Any, ...] = ()
    ) -> ExecutionResult:
        """Run one statement, returning any result rows and the affected count."""
        with self.get_connection() as connection:
            cursor = connection.execute(query, params)
            rows = (
                [dict(row) for row in cursor.fetchall()]
                if cursor.description is not None
                else []
            )
            connection.commit()
            return ExecutionResult(rows=rows, affected=cursor.rowcount)

    @log_call("persistence")
    def execute_script(self, script: str) -> None:
        """Run a multi-statement SQL script (e.g. DDL or a bulk data dump)."""
        with self.get_connection() as connection:
            connection.executescript(script)
            connection.commit()
