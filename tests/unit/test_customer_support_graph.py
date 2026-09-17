"""Unit tests for the customer support agent's LangGraph graph."""

from pathlib import Path
from typing import Any

import pytest

from agentic_commons.config import get_settings
from agentic_commons.llm.types import LLMResponse, Message, ToolCall
from agentic_commons.persistence.database import Database
from customer_support_agent.core.graph import run_agent
from customer_support_agent.core.repositories import OrdersRepository, UsersRepository
from customer_support_agent.core.schema import ensure_schema


class _ScriptedClient:
    """Fake LLMClient that returns a scripted sequence of responses.

    Each entry is either an LLMResponse (returned as-is) or a callable
    taking the messages passed to generate() and returning an LLMResponse,
    for assertions on what the graph actually sent.
    """

    def __init__(self, script: list[Any]) -> None:
        self._script = script
        self.calls: list[list[Message]] = []

    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        self.calls.append(messages)
        step = self._script[len(self.calls) - 1]
        return step(messages) if callable(step) else step

    def generate_stream(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def health_check(self) -> Any:
        raise NotImplementedError


def _seeded_db(tmp_path: Path) -> Database:
    database = Database(db_path=tmp_path / "test.db")
    ensure_schema(database)
    UsersRepository(database).insert_user("user-1", "alice@example.com")
    OrdersRepository(database).insert_order(
        "small", "user-1", "delivered", "none", 29.99
    )
    OrdersRepository(database).insert_order(
        "large", "user-1", "delivered", "none", 149.50
    )
    OrdersRepository(database).insert_order(
        "refunded", "user-1", "delivered", "completed", 42.00
    )
    return database


def test_query_with_no_tool_call_returns_direct_answer(tmp_path: Path) -> None:
    db = _seeded_db(tmp_path)
    client = _ScriptedClient([LLMResponse(content="Hello, how can I help?")])

    result = run_agent("hi", llm_client=client, database=db)

    assert result == "Hello, how can I help?"


def test_query_processes_a_small_refund_directly(tmp_path: Path) -> None:
    db = _seeded_db(tmp_path)
    client = _ScriptedClient(
        [
            LLMResponse(
                tool_calls=[ToolCall(id="c1", name="process_refund", arguments='{"order_id": "small"}')]
            ),
            LLMResponse(content="Refunded!"),
        ]
    )

    result = run_agent("refund order small", llm_client=client, database=db)

    assert result == "Refunded!"
    order = db.execute(
        "SELECT refund_status FROM orders WHERE order_id = ?", ("small",)
    ).rows[0]
    assert order["refund_status"] == "completed"


def test_query_queues_large_refund_for_approval(tmp_path: Path) -> None:
    db = _seeded_db(tmp_path)
    client = _ScriptedClient(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(id="c1", name="queue_for_approval", arguments='{"order_id": "large"}')
                ]
            ),
            LLMResponse(content="Queued for approval."),
        ]
    )

    result = run_agent("refund order large", llm_client=client, database=db)

    assert result == "Queued for approval."
    order = db.execute(
        "SELECT refund_status FROM orders WHERE order_id = ?", ("large",)
    ).rows[0]
    assert order["refund_status"] == "pending_review"
    queued = db.execute("SELECT * FROM hitl_queue").rows
    assert len(queued) == 1
    assert queued[0]["order_id"] == "large"


def test_tool_result_is_fed_back_to_the_model(tmp_path: Path) -> None:
    db = _seeded_db(tmp_path)
    seen_tool_content: list[str] = []

    def second_call(messages: list[Message]) -> LLMResponse:
        tool_message = next(m for m in messages if m.role == "tool")
        seen_tool_content.append(tool_message.content)
        return LLMResponse(content="done")

    client = _ScriptedClient(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(id="c1", name="get_order_status", arguments='{"order_id": "small"}')
                ]
            ),
            second_call,
        ]
    )

    run_agent("status of order small", llm_client=client, database=db)

    assert len(seen_tool_content) == 1
    assert "delivered" in seen_tool_content[0]


def test_max_tool_iterations_forces_a_final_text_answer(tmp_path: Path) -> None:
    db = _seeded_db(tmp_path)
    max_iterations = get_settings().customer_support_agent.max_tool_iterations

    def always_wants_a_tool(messages: list[Message]) -> LLMResponse:
        return LLMResponse(
            content="",
            tool_calls=[
                ToolCall(id="c", name="get_order_status", arguments='{"order_id": "small"}')
            ],
        )

    script = [always_wants_a_tool] * (max_iterations + 1) + [
        LLMResponse(content="Here is my final answer.")
    ]
    client = _ScriptedClient(script)

    result = run_agent("keep asking forever", llm_client=client, database=db)

    assert result == "Here is my final answer."
    # one initial + max_iterations tool-driven + one forced finalize call
    assert len(client.calls) == max_iterations + 2


def test_gatekeeper_blocks_process_refund_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(get_settings().gatekeeper, "enabled", True)
    db = _seeded_db(tmp_path)
    client = _ScriptedClient(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(id="c1", name="process_refund", arguments='{"order_id": "small"}')
                ]
            ),
            LLMResponse(content="done"),
        ]
    )

    run_agent("refund order small", llm_client=client, database=db)

    order = db.execute(
        "SELECT refund_status FROM orders WHERE order_id = ?", ("small",)
    ).rows[0]
    # Gatekeeper is still a Phase 0 stub that always denies — the refund
    # must NOT have gone through while gatekeeper is enabled.
    assert order["refund_status"] == "none"


def test_pii_check_runs_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(get_settings().security.pii, "enabled", True)
    db = _seeded_db(tmp_path)
    client = _ScriptedClient([LLMResponse(content="ok")])

    result = run_agent("hi", llm_client=client, database=db)

    assert result == "ok"  # graph still completes with the check turned on
