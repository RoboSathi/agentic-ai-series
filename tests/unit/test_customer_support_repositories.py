"""Unit tests for the customer support agent's schema and repositories."""

from pathlib import Path

from agentic_commons.persistence.database import Database
from customer_support_agent.core.repositories import (
    HitlQueueRepository,
    OrdersRepository,
    UsersRepository,
)
from customer_support_agent.core.schema import ensure_schema


def _seeded_db(tmp_path: Path) -> Database:
    database = Database(db_path=tmp_path / "test.db")
    ensure_schema(database)
    UsersRepository(database).insert_user("user-1", "alice@example.com", "555-0100")
    OrdersRepository(database).insert_order(
        "12345", "user-1", "delivered", "none", 29.99
    )
    return database


def test_ensure_schema_is_idempotent(tmp_path: Path) -> None:
    database = Database(db_path=tmp_path / "test.db")
    ensure_schema(database)
    ensure_schema(database)  # must not raise on the second call


def test_get_order_returns_row(tmp_path: Path) -> None:
    database = _seeded_db(tmp_path)
    orders = OrdersRepository(database)

    order = orders.get_order("12345")

    assert order is not None
    assert order["user_id"] == "user-1"
    assert order["order_status"] == "delivered"
    assert order["refund_status"] == "none"
    assert order["amount"] == 29.99


def test_get_order_returns_none_when_missing(tmp_path: Path) -> None:
    database = _seeded_db(tmp_path)
    orders = OrdersRepository(database)

    assert orders.get_order("nonexistent") is None


def test_set_refund_status_updates_the_row(tmp_path: Path) -> None:
    database = _seeded_db(tmp_path)
    orders = OrdersRepository(database)

    orders.set_refund_status("12345", "completed")

    order = orders.get_order("12345")
    assert order is not None
    assert order["refund_status"] == "completed"


def test_get_user_returns_row(tmp_path: Path) -> None:
    database = _seeded_db(tmp_path)
    users = UsersRepository(database)

    user = users.get_user("user-1")

    assert user is not None
    assert user["email"] == "alice@example.com"


def test_get_user_returns_none_when_missing(tmp_path: Path) -> None:
    database = _seeded_db(tmp_path)
    users = UsersRepository(database)

    assert users.get_user("nonexistent") is None


def test_hitl_queue_enqueue_inserts_a_pending_row(tmp_path: Path) -> None:
    database = _seeded_db(tmp_path)
    hitl_queue = HitlQueueRepository(database)

    hitl_queue.enqueue("12345", "user-1", 149.50)

    result = database.execute("SELECT * FROM hitl_queue")
    assert len(result.rows) == 1
    assert result.rows[0]["order_id"] == "12345"
    assert result.rows[0]["requested_amount"] == 149.50
    assert result.rows[0]["status"] == "pending"
