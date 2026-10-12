"""Shared analytical observability storage.

The agent's relational/business state stays in the agent-owned SQLite database.
Observability is different: it is append-heavy, cross-agent and queried for
analytics rather than live business transactions, so it gets a shared DuckDB
store.  The store is intentionally tiny and generic; observability components
own their tables, while the store owns the connection and serialization of
access.
"""

from __future__ import annotations

import json
import tempfile
import threading
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from agent_platform.config import get_settings
from agent_platform.core.persistence.types import ExecutionResult


class DuckDbObservabilityStore:
    """Thread-safe, process-local DuckDB store shared by all agents.

    DuckDB is imported lazily so the rest of the common package remains
    importable in environments that have not installed the analytics optional
    dependency yet. A runtime attempt to construct the store fails with a
    precise installation message instead of an opaque import error elsewhere.
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        try:
            import duckdb
        except ImportError as exc:  # pragma: no cover - exercised in packaging/runtime envs
            raise RuntimeError(
                "DuckDB observability is enabled but the 'duckdb' package is not installed. "
                "Install the project dependencies before starting an observed agent."
            ) from exc

        configured = (
            db_path if db_path is not None else get_settings().analytics.recording.resolved_db_path
        )
        self._db_path = str(configured)
        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._connection = duckdb.connect(self._db_path)

    @property
    def db_path(self) -> str:
        """Resolved DuckDB path."""
        return self._db_path

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> ExecutionResult:
        """Run one statement and return rows/affected count."""
        with self._lock:
            cursor = self._connection.execute(query, params)
            rows = self._rows(cursor)
            self._connection.commit()
            affected = cursor.rowcount if cursor.rowcount is not None else -1
            return ExecutionResult(rows=rows, affected=affected)

    def execute_many(self, query: str, seq_of_params: list[tuple[Any, ...]]) -> ExecutionResult:
        """Run one prepared statement against many rows."""
        with self._lock:
            cursor = self._connection.executemany(query, seq_of_params)
            self._connection.commit()
            affected = cursor.rowcount if cursor.rowcount is not None else len(seq_of_params)
            return ExecutionResult(rows=[], affected=affected)

    def bulk_insert(
        self, table: str, columns: Mapping[str, str], rows: Sequence[Mapping[str, Any]]
    ) -> int:
        """Append many rows in one statement (`columns`: name → DuckDB type).

        `executemany` inserts row by row; for a backfill of hundreds of thousands of
        rows the rows are staged as newline-delimited JSON and read in one scan.
        """
        if not rows:
            return 0
        names = ", ".join(f'"{name}"' for name in columns)
        types = ", ".join(f"'{name}': '{kind}'" for name, kind in columns.items())
        with tempfile.TemporaryDirectory() as folder:
            staged = Path(folder) / f"{table}.jsonl"
            with staged.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps({name: row.get(name) for name in columns}, default=str))
                    handle.write("\n")
            literal = str(staged).replace("'", "''")
            self.execute_script(
                f"INSERT INTO {table} ({names}) SELECT {names} FROM read_json('{literal}', "
                f"format = 'newline_delimited', columns = {{{types}}})"
            )
        return len(rows)

    def execute_script(self, script: str) -> None:
        """Execute a DDL/data script."""
        with self._lock:
            self._connection.execute(script)
            self._connection.commit()

    def query_one(self, query: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        """Return the first row, if any."""
        rows = self.execute(query, params).rows
        return rows[0] if rows else None

    def iter_rows(self, query: str, params: tuple[Any, ...] = ()) -> Iterator[dict[str, Any]]:
        """Yield result rows in order."""
        yield from self.execute(query, params).rows

    def close(self) -> None:
        """Close the DuckDB connection."""
        with self._lock:
            self._connection.close()

    @staticmethod
    def _rows(cursor: Any) -> list[dict[str, Any]]:
        """Convert DuckDB's cursor output to the repository-style row mapping."""
        if cursor.description is None:
            return []
        columns = [str(column[0]) for column in cursor.description]
        return [dict(zip(columns, row, strict=False)) for row in cursor.fetchall()]


def create_observability_store() -> DuckDbObservabilityStore:
    """Create the configured shared observability store."""
    return DuckDbObservabilityStore()
