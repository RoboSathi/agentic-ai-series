"""Integration tests for the customer support agent's LangGraph graph.

Exercises the full compiled graph end-to-end (schema creation, tool
registration, the LLM<->tools loop, and the toggle-gated pipeline) with a
fake LLM client and a real temporary SQLite database — no network calls.
"""

from pathlib import Path
from typing import Any

from agentic_commons.config import get_settings
from agentic_commons.llm.types import LLMResponse, Message, ToolCall
from agentic_commons.persistence.database import Database
from customer_support_agent.core.graph import build_graph, run_agent
from customer_support_agent.core.repositories import OrdersRepository, UsersRepository
from customer_support_agent.core.schema import ensure_schema


class _ScriptedClient:
    def __init__(self, script: list[LLMResponse]) -> None:
        self._script = script
        self.call_count = 0

    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        response = self._script[self.call_count]
        self.call_count += 1
        return response

    def generate_stream(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def health_check(self) -> Any:
        raise NotImplementedError


def _seeded_db(tmp_path: Path) -> Database:
    database = Database(db_path=tmp_path / "test.db")
    ensure_schema(database)
    UsersRepository(database).insert_user("user-1", "alice@example.com")
    OrdersRepository(database).insert_order(
        "12345", "user-1", "delivered", "none", 29.99
    )
    return database


def test_build_graph_compiles_and_is_invokable(tmp_path: Path) -> None:
    """Test that build_graph returns a compiled, invokable graph."""
    db = _seeded_db(tmp_path)
    client = _ScriptedClient([LLMResponse(content="hello")])

    compiled = build_graph(llm_client=client, database=db)

    assert hasattr(compiled, "invoke")


def test_run_agent_end_to_end_with_a_real_temp_database(tmp_path: Path) -> None:
    """Test a full ask -> tool call -> tool result -> answer round trip."""
    db = _seeded_db(tmp_path)
    client = _ScriptedClient(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(id="c1", name="get_order_status", arguments='{"order_id": "12345"}')
                ]
            ),
            LLMResponse(content="Your order is delivered."),
        ]
    )

    result = run_agent("status of order 12345", llm_client=client, database=db)

    assert result == "Your order is delivered."
    assert client.call_count == 2


def test_run_agent_creates_schema_on_a_fresh_database(tmp_path: Path) -> None:
    """Test that run_agent works against a brand-new, unmigrated database file."""
    db = Database(db_path=tmp_path / "brand_new.db")
    client = _ScriptedClient([LLMResponse(content="hi there")])

    result = run_agent("hello", llm_client=client, database=db)

    assert result == "hi there"


def test_log_dir_created() -> None:
    """Test that log directory is created by logging setup."""
    settings = get_settings()
    log_dir = Path(settings.logging.log_dir)

    assert log_dir.exists() or not settings.logging.file_enabled
