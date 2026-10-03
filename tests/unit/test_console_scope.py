"""Console commands show the verified customer's records, and nobody else's.

`/queue` and `/memory` were operator views of everything: verified as one
customer, the console listed every customer's pending refunds and let
`/memory customer <id>` read anyone's memory. Scoped by the session's
verified subject (`ConsoleScope`), they now show only that customer's.
"""

from typing import Any

import pytest

from agent_platform.commands import ConsoleScope, MemoryCommand
from agent_platform.memory.working import WorkingMemoryStore
from agent_platform.persistence import SqliteConnectionPool
from agent_platform.primitives.money import Money
from customer_support_agent.console.commands import QueueCommand
from customer_support_agent.domain.models import HitlRequest
from customer_support_agent.persistence.repositories import HitlQueueRepository


def _scope(verified: dict[str, str], session: list[str]) -> ConsoleScope:
    """A console on `session[0]`, where `verified` says who proved themselves in which session."""
    return ConsoleScope(lambda: session[0], verified.get)


@pytest.fixture
def queued(seeded_pool: SqliteConnectionPool) -> SqliteConnectionPool:
    """One pending refund for customer 1 (order 10001) and one for customer 2 (order 10003)."""
    queue = HitlQueueRepository(seeded_pool)
    queue.save(HitlRequest(order_id=10001, user_id=1, requested_amount=Money.parse("20.00")))
    queue.save(HitlRequest(order_id=10003, user_id=2, requested_amount=Money.parse("45.00")))
    return seeded_pool


def _queue(pool: SqliteConnectionPool, scope: ConsoleScope) -> QueueCommand:
    command = QueueCommand(pool)
    command.bind_scope(scope)
    return command


def _id_for(pool: SqliteConnectionPool, user_id: int) -> int:
    [request] = HitlQueueRepository(pool).pending_for_user(user_id)
    assert request.id is not None
    return request.id


class TestTheQueueIsTheVerifiedCustomersOwn:
    async def test_it_lists_only_their_requests(self, queued: SqliteConnectionPool) -> None:
        result = await _queue(queued, _scope({"s": "1"}, ["s"])).run([])
        assert "1 pending" in result.title
        assert [row.cells[1] for row in result.rows] == ["10001"]
        assert "customer" not in result.headers

    async def test_another_customers_request_cannot_be_decided(
        self, queued: SqliteConnectionPool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        decided: list[Any] = []
        monkeypatch.setattr(
            "customer_support_agent.graph.builder.build_review_service",
            lambda pool=None, audit=None: decided.append(pool),
        )
        theirs = _id_for(queued, 2)
        result = await _queue(queued, _scope({"s": "1"}, ["s"])).run(["approve", str(theirs)])
        assert decided == []
        assert any("not pending" in line for line in result.lines)

    async def test_nobody_verified_means_nothing_is_shown(
        self, queued: SqliteConnectionPool
    ) -> None:
        result = await _queue(queued, _scope({}, ["s"])).run([])
        assert not result.ok and "Verify first" in " ".join(result.lines)

    async def test_unscoped_it_is_the_whole_queue(self, queued: SqliteConnectionPool) -> None:
        result = await QueueCommand(queued).run([])
        assert "2 pending" in result.title and "customer" in result.headers


class TestMemoryIsTheVerifiedCustomersOwn:
    async def test_someone_elses_memory_is_refused(self, seeded_pool: SqliteConnectionPool) -> None:
        command = MemoryCommand(seeded_pool)
        command.bind_scope(_scope({"s": "1"}, ["s"]))
        result = await command.run(["customer", "2"])
        assert not result.ok and "only the verified account" in " ".join(result.lines)

    async def test_nobody_verified_means_nothing_is_shown(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        command = MemoryCommand(seeded_pool)
        command.bind_scope(_scope({}, ["s"]))
        result = await command.run([])
        assert not result.ok and "Verify first" in " ".join(result.lines)

    async def test_working_memory_follows_the_console_session(self) -> None:
        store = WorkingMemoryStore()
        store.get("second").set("verified_customer", "Ada (account #1)")
        session = ["first"]
        command = MemoryCommand(working_memory=store, session_id="first")
        command.bind_scope(_scope({"second": "1"}, session))
        session[0] = "second"
        result = await command.run(["working"])
        assert [row.cells[0] for row in result.rows] == ["verified_customer"]
