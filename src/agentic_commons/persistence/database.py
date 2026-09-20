"""SQLite access — the only module that opens a database connection (LLD §4)."""

import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterator, Protocol, TypeVar, runtime_checkable

from agentic_commons.config import get_settings
from agentic_commons.logging import logged
from agentic_commons.persistence.types import ExecutionResult

T = TypeVar("T")

IN_MEMORY = ":memory:"


@runtime_checkable
class Repository(Protocol[T]):
    """Aggregate-oriented storage interface (LLD §4).

    Concrete repositories live with their owning agent/domain and depend on
    `SqliteConnectionPool`, never on `sqlite3` directly. Platform-owned
    common state uses the higher-level `PlatformStore` contract.
    """

    def get(self, id: str) -> T | None: ...

    def save(self, entity: T) -> None: ...

    def query(self, **filters: Any) -> list[T]: ...


class SqliteConnectionPool:
    """Owns one SQLite connection, shared safely across threads.

    A single long-lived connection (rather than one per call) is what makes
    `:memory:` usable as a real test fixture — a fresh connection to
    `:memory:` would silently get an empty database every time.
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        configured = db_path if db_path is not None else get_settings().persistence.resolved_db_path
        self._db_path = str(configured)
        if self._db_path != IN_MEMORY:
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection: sqlite3.Connection | None = None

    @property
    def db_path(self) -> str:
        return self._db_path

    def connection(self) -> sqlite3.Connection:
        """The shared connection, opened on first use."""
        with self._lock:
            if self._connection is None:
                self._connection = sqlite3.connect(self._db_path, check_same_thread=False)
                self._connection.row_factory = sqlite3.Row
                self._connection.execute("PRAGMA foreign_keys = ON")
            return self._connection

    def close(self) -> None:
        """Close the shared connection (tests and shutdown paths use this)."""
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    @logged("persistence")
    def execute(self, query: str, params: tuple[Any, ...] = ()) -> ExecutionResult:
        """Run one statement, returning any rows and the affected count."""
        with self._lock:
            connection = self.connection()
            cursor = connection.execute(query, params)
            rows = (
                [dict(row) for row in cursor.fetchall()] if cursor.description is not None else []
            )
            connection.commit()
            return ExecutionResult(rows=rows, affected=cursor.rowcount)

    @logged("persistence")
    def execute_many(self, query: str, seq_of_params: list[tuple[Any, ...]]) -> ExecutionResult:
        """Run one statement against many parameter sets in a single transaction."""
        with self._lock:
            connection = self.connection()
            cursor = connection.executemany(query, seq_of_params)
            connection.commit()
            return ExecutionResult(rows=[], affected=cursor.rowcount)

    @logged("persistence")
    def execute_script(self, script: str) -> None:
        """Run a multi-statement SQL script (DDL or a bulk data dump)."""
        with self._lock:
            connection = self.connection()
            connection.executescript(script)
            connection.commit()

    def query_one(self, query: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        """First row of a query, or None."""
        rows = self.execute(query, params).rows
        return rows[0] if rows else None

    def iter_rows(self, query: str, params: tuple[Any, ...] = ()) -> Iterator[dict[str, Any]]:
        """Stream result rows without materializing the whole set twice."""
        yield from self.execute(query, params).rows
