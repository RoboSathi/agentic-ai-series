"""Creating this agent's tables from the single source of truth.

The DDL lives in `schema.sql` beside this module and is read from disk
rather than duplicated here: two copies of a schema drift, and the copy
in the test fixture is always the one that drifts.
"""

from pathlib import Path

from agentic_commons.persistence import SqliteConnectionPool

_SCHEMA_FILE = Path(__file__).resolve().parent / "schema.sql"


def ensure_schema(pool: SqliteConnectionPool) -> None:
    """Create the domain tables from the single source of truth."""
    pool.execute_script(_SCHEMA_FILE.read_text())
