"""Creating this agent's tables from the single source of truth.

The DDL lives in `schema.sql` beside this module and is read from disk
rather than duplicated here: two copies of a schema drift, and the copy
in the test fixture is always the one that drifts.

A database built before reply reviews shares the queue has a `hitl_queue`
whose order and amount are NOT NULL. SQLite cannot relax a constraint in
place, so that table is rebuilt once, rows kept, the first time it is
opened by this version.
"""

from pathlib import Path

from agent_platform.persistence import SqliteConnectionPool

_SCHEMA_FILE = Path(__file__).resolve().parent / "schema.sql"
# The column a queue built by this version has and an older one lacks.
_REPLY_REVIEW_COLUMN = "detail"
_OLD_QUEUE = "hitl_queue_before_reply_reviews"
_QUEUE_COLUMNS = "id, order_id, user_id, requested_amount_cents, action, status, created_at"


def ensure_schema(pool: SqliteConnectionPool) -> None:
    """Create the domain tables from the single source of truth, upgrading an older queue."""
    schema = _SCHEMA_FILE.read_text()
    if not _queue_predates_reply_reviews(pool):
        pool.execute_script(schema)
        return
    # One script, one transaction: a rebuild that stops halfway leaves the old table in place.
    # Rows are copied as they are, so the copy does not re-check their references, which is
    # set outside the transaction and put back as it was.
    enforced = (pool.query_one("PRAGMA foreign_keys") or {}).get("foreign_keys", 0)
    pool.execute_script(
        "PRAGMA foreign_keys = OFF;"
        "BEGIN;"
        f"ALTER TABLE hitl_queue RENAME TO {_OLD_QUEUE};"
        "DROP INDEX IF EXISTS idx_hitl_status;"
        f"{schema}"
        f"INSERT INTO hitl_queue ({_QUEUE_COLUMNS}) SELECT {_QUEUE_COLUMNS} FROM {_OLD_QUEUE};"
        f"DROP TABLE {_OLD_QUEUE};"
        "COMMIT;"
        f"PRAGMA foreign_keys = {int(enforced)};"
    )


def _queue_predates_reply_reviews(pool: SqliteConnectionPool) -> bool:
    columns = {row["name"] for row in pool.iter_rows("PRAGMA table_info(hitl_queue)")}
    return bool(columns) and _REPLY_REVIEW_COLUMN not in columns
