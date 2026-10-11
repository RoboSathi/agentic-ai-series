"""Creating this agent's tables from the single source of truth.

The DDL lives in `schema.sql` beside this module and is read from disk
rather than duplicated here: two copies of a schema drift, and the copy
in the test fixture is always the one that drifts.

A database built before the platform's one review queue (ADR 0011) has a
`hitl_queue` table of its own. The first time this version opens it, its
rows are moved into the review queue as items, and the table is dropped.
"""

from pathlib import Path

from agent_platform.config import get_settings
from agent_platform.persistence import SqliteConnectionPool
from agent_platform.primitives.money import Money
from agent_platform.review import ReviewQueue
from customer_support_agent.config import get_agent_settings
from customer_support_agent.domain.models import HitlRequest, ReplyReview
from customer_support_agent.persistence.repositories import HitlQueueRepository, UsersRepository

_SCHEMA_FILE = Path(__file__).resolve().parent / "schema.sql"
# The column a queue built by this version has and an older one lacks.
# The table escalations lived in before the platform's review queue.
_OLD_QUEUE = "hitl_queue"
# Order columns added after databases were already built. `CREATE TABLE IF NOT
# EXISTS` leaves an existing table as it was, so each is added in place, once.
_ADDED_ORDER_COLUMNS = {"carrier_note": "TEXT", "refund_settles_at": "TEXT"}
# Refund statuses no code path produces any more, and what an older row means now.
# `approved` was only ever seeded: a reviewer's approval issues the refund, so a seeded
# approved refund is one that has paid out and settled.
_RETIRED_REFUND_STATUSES = {"approved": "completed"}


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
    _add_missing_columns(pool, "orders", _ADDED_ORDER_COLUMNS)
    _retire_refund_statuses(pool)
    pool.execute_script(_SCHEMA_FILE.read_text())
    # The platform's review queue, where this agent's escalations are items (ADR 0011).
    ReviewQueue(pool, agent_id=get_settings().agent.id)
    _move_old_queue(pool)


def _move_old_queue(pool: SqliteConnectionPool) -> None:
    """Move an older database's `hitl_queue` rows into the review queue, then drop it."""
    present = {row["name"] for row in pool.iter_rows(f"PRAGMA table_info({_OLD_QUEUE})")}
    if not present:
        return
    queue = HitlQueueRepository(pool)
    for row in pool.iter_rows(f"SELECT * FROM {_OLD_QUEUE} ORDER BY id"):
        cents = row.get("requested_amount_cents")
        detail = row.get("detail")
        queue.save(
            HitlRequest(
                order_id=row.get("order_id"),
                user_id=int(row["user_id"]),
                requested_amount=None if cents is None else Money.from_cents(int(cents)),
                action=row.get("action") or "refund",
                status=row["status"],
                created_at=str(row["created_at"]),
                session_id=row.get("session_id"),
                detail=None if not detail else ReplyReview.model_validate_json(detail),
                reason=str(row.get("reason") or ""),
            )
        )
    pool.execute_script(f"DROP INDEX IF EXISTS idx_hitl_status; DROP TABLE {_OLD_QUEUE};")


def _add_missing_columns(pool: SqliteConnectionPool, table: str, columns: dict[str, str]) -> None:
    """Add any column this version declares and an older table lacks."""
    present = {row["name"] for row in pool.iter_rows(f"PRAGMA table_info({table})")}
    if not present:
        return
    for column, declaration in columns.items():
        if column not in present:
            pool.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")


def _retire_refund_statuses(pool: SqliteConnectionPool) -> None:
    """Rewrite any order still carrying a refund status this version no longer has."""
    if not list(pool.iter_rows("PRAGMA table_info(orders)")):
        return
    for retired, current in _RETIRED_REFUND_STATUSES.items():
        pool.execute(
            "UPDATE orders SET refund_status = ? WHERE refund_status = ?", (current, retired)
        )
