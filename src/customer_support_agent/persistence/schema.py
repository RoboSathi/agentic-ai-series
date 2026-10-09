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
from customer_support_agent.config import get_agent_settings
from customer_support_agent.persistence.repositories import UsersRepository

_SCHEMA_FILE = Path(__file__).resolve().parent / "schema.sql"
# The column a queue built by this version has and an older one lacks.
_REPLY_REVIEW_COLUMN = "detail"
_OLD_QUEUE = "hitl_queue_before_reply_reviews"
_QUEUE_COLUMNS = "id, order_id, user_id, requested_amount_cents, action, status, created_at"
# Order columns added after databases were already built. `CREATE TABLE IF NOT
# EXISTS` leaves an existing table as it was, so each is added in place, once.
_ADDED_ORDER_COLUMNS = {"carrier_note": "TEXT", "refund_settles_at": "TEXT"}


class AgentDatabaseNotSeededError(RuntimeError):
    """The agent's own database has its tables but no customers in them."""


def require_seeded(pool: SqliteConnectionPool, path: Path) -> None:
    """Refuse a database with no customers, rather than answer every code as unknown.

    `ensure_schema` makes an empty file a valid database, so a deleted one comes
    back as tables with nobody in them: the agent would start, look healthy, and
    reject every customer. Only the agent's own file is checked; a pool a caller
    injects is that caller's fixture.
    """
    if UsersRepository(pool).count() == 0:
        message = get_agent_settings().unseeded_database.format(path=path)
        raise AgentDatabaseNotSeededError(message)


def ensure_schema(pool: SqliteConnectionPool) -> None:
    """Create the domain tables from the single source of truth, upgrading older tables."""
    schema = _SCHEMA_FILE.read_text()
    _add_missing_order_columns(pool)
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


def _add_missing_order_columns(pool: SqliteConnectionPool) -> None:
    """Add any order column this version declares and an older `orders` table lacks."""
    present = {row["name"] for row in pool.iter_rows("PRAGMA table_info(orders)")}
    if not present:
        return
    for column, declaration in _ADDED_ORDER_COLUMNS.items():
        if column not in present:
            pool.execute(f"ALTER TABLE orders ADD COLUMN {column} {declaration}")
