"""Unit tests for the customer support agent's LLM-callable tools."""

import json
from pathlib import Path

from agentic_commons.persistence.database import Database
from agentic_commons.tools.registry import ToolRegistry
from customer_support_agent.core.repositories import OrdersRepository, UsersRepository
from customer_support_agent.core.schema import ensure_schema
from customer_support_agent.core.tools import build_tool_registry


def _seeded_registry(tmp_path: Path) -> tuple[ToolRegistry, Database]:
    database = Database(db_path=tmp_path / "test.db")
    ensure_schema(database)
    UsersRepository(database).insert_user("user-1", "alice@example.com")
    OrdersRepository(database).insert_order(
        "order-small", "user-1", "delivered", "none", 29.99
    )
    OrdersRepository(database).insert_order(
        "order-large", "user-1", "delivered", "none", 149.50
    )
    OrdersRepository(database).insert_order(
        "order-refunded", "user-1", "delivered", "completed", 42.00
    )
    return build_tool_registry(database), database


def test_get_order_status_returns_status(tmp_path: Path) -> None:
    registry, _ = _seeded_registry(tmp_path)

    result = registry.invoke("get_order_status", {"order_id": "order-small"})

    assert result.success is True
    assert json.loads(result.output) == {
        "order_id": "order-small",
        "order_status": "delivered",
    }


def test_get_order_status_unknown_order_fails(tmp_path: Path) -> None:
    registry, _ = _seeded_registry(tmp_path)

    result = registry.invoke("get_order_status", {"order_id": "nonexistent"})

    assert result.success is False


def test_get_refund_status_returns_amount_and_status(tmp_path: Path) -> None:
    registry, _ = _seeded_registry(tmp_path)

    result = registry.invoke("get_refund_status", {"order_id": "order-small"})

    assert result.success is True
    assert json.loads(result.output) == {
        "order_id": "order-small",
        "refund_status": "none",
        "amount": 29.99,
    }


def test_process_refund_completes_and_updates_status(tmp_path: Path) -> None:
    registry, database = _seeded_registry(tmp_path)

    result = registry.invoke("process_refund", {"order_id": "order-small"})

    assert result.success is True
    assert "29.99" in result.output
    order = database.execute(
        "SELECT refund_status FROM orders WHERE order_id = ?", ("order-small",)
    ).rows[0]
    assert order["refund_status"] == "completed"


def test_process_refund_rejects_already_refunded_order(tmp_path: Path) -> None:
    registry, _ = _seeded_registry(tmp_path)

    result = registry.invoke("process_refund", {"order_id": "order-refunded"})

    assert result.success is False
    assert "already been refunded" in result.output


def test_queue_for_approval_marks_pending_review_and_enqueues(tmp_path: Path) -> None:
    registry, database = _seeded_registry(tmp_path)

    result = registry.invoke("queue_for_approval", {"order_id": "order-large"})

    assert result.success is True
    order = database.execute(
        "SELECT refund_status FROM orders WHERE order_id = ?", ("order-large",)
    ).rows[0]
    assert order["refund_status"] == "pending_review"
    queued = database.execute("SELECT * FROM hitl_queue").rows
    assert len(queued) == 1
    assert queued[0]["order_id"] == "order-large"
    assert queued[0]["requested_amount"] == 149.50


def test_queue_for_approval_rejects_already_refunded_order(tmp_path: Path) -> None:
    registry, _ = _seeded_registry(tmp_path)

    result = registry.invoke("queue_for_approval", {"order_id": "order-refunded"})

    assert result.success is False


def test_registry_exposes_all_four_tools_to_openai_schema(tmp_path: Path) -> None:
    registry, _ = _seeded_registry(tmp_path)

    names = {tool["function"]["name"] for tool in registry.to_openai_tools()}

    assert names == {
        "get_order_status",
        "get_refund_status",
        "process_refund",
        "queue_for_approval",
    }
