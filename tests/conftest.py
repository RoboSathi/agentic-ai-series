"""Shared test fixtures.

Two principles run through this file.

First, no test touches a real model or a real database file. The LLM is
scripted, so a test asserts on the *harness's* behaviour rather than on
whatever the model felt like producing that afternoon; the database is
`:memory:`, so tests cannot leak into each other or into the dev data.

Second, config is restored after any test that changes it. The settings
object is a process-wide singleton, so a test that flips a toggle and
leaves it flipped fails a different test later, in a different file,
for no visible reason.
"""

import json
from collections.abc import Iterator
from typing import Any

import pytest

from agentic_commons.config import get_settings, reload_settings
from agentic_commons.llm.client import LLMClient
from agentic_commons.llm.types import (
    CompletionRequest,
    CompletionResult,
    HealthCheckResult,
    ModelToolCall,
    TokenUsage,
)
from agentic_commons.persistence import IN_MEMORY, SqliteConnectionPool
from customer_support_agent.core.schema import Order, User, ensure_schema


class ScriptedLLM(LLMClient):
    """An LLM that returns a fixed script.

    Every graph test needs the model to do something specific — call
    this tool, then that one, then answer. A real model does that
    *usually*, which is not a basis for a CI gate.
    """

    def __init__(self, script: list[dict[str, Any]] | None = None) -> None:
        self.script = script or [{"content": "ok"}]
        self.requests: list[CompletionRequest] = []
        self._index = 0

    @property
    def tier(self) -> str:
        return "scripted"

    async def health_check(self) -> HealthCheckResult:
        return HealthCheckResult(healthy=True, latency_ms=0.0)

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        """Return the next scripted step, repeating the last one if exhausted."""
        self.requests.append(request)
        step = self.script[min(self._index, len(self.script) - 1)]
        self._index += 1
        return CompletionResult(
            content=step.get("content", ""),
            tool_calls=step.get("tool_calls"),
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            tier="scripted",
            cost_usd=0.0001,
        )


def tool_call(name: str, **arguments: Any) -> list[ModelToolCall]:
    """Build a model tool call for a script step."""
    return [ModelToolCall(id=name, name=name, arguments=json.dumps(arguments))]


@pytest.fixture
def pool() -> Iterator[SqliteConnectionPool]:
    """An in-memory database, isolated per test."""
    connection = SqliteConnectionPool(IN_MEMORY)
    yield connection
    connection.close()


@pytest.fixture
def seeded_pool(pool: SqliteConnectionPool) -> SqliteConnectionPool:
    """A database with the domain schema and a handful of known orders.

    Fixed values, not generated ones: a test that asserts "$32 is
    auto-approved" needs a $32 order to exist by construction.
    """
    ensure_schema(pool)
    from customer_support_agent.core.repositories import OrdersRepository, UsersRepository

    users = UsersRepository(pool)
    orders = OrdersRepository(pool)
    users.save(User(user_id="u1", created_at="2026-01-01", email="ada@example.com"))
    users.save(User(user_id="u2", created_at="2026-01-02", email="grace@example.com"))
    orders.save(
        Order(
            order_id="10001",
            user_id="u1",
            timestamp="2026-02-01",
            order_status="delivered",
            refund_status="none",
            amount=32.00,
        )
    )
    orders.save(
        Order(
            order_id="10002",
            user_id="u1",
            timestamp="2026-02-02",
            order_status="delivered",
            refund_status="none",
            amount=210.00,
        )
    )
    orders.save(
        Order(
            order_id="10003",
            user_id="u2",
            timestamp="2026-02-03",
            order_status="delivered",
            refund_status="completed",
            amount=45.00,
        )
    )
    return pool


@pytest.fixture
def scripted_llm() -> ScriptedLLM:
    """An LLM that answers without calling anything."""
    return ScriptedLLM()


@pytest.fixture
def restore_settings() -> Iterator[None]:
    """Undo any config mutation a test performs."""
    yield
    reload_settings()


@pytest.fixture
def security_on(restore_settings: None) -> Iterator[None]:
    """Turn on the content-level security checks that ship off by default."""
    settings = get_settings().security
    settings.pii.enabled = True
    settings.prompt_injection.enabled = True
    settings.content_moderation.enabled = True
    yield
