"""Console commands show the verified customer's records, and nobody else's.

`/queue` and `/memory` were operator views of everything: verified as one
customer, the console listed every customer's pending refunds and let
`/memory customer <id>` read anyone's memory. Scoped by the session's
verified subject (`ConsoleScope`), they now show only that customer's.
"""

from typing import Any

import pytest

from agent_platform.core.persistence import SqliteConnectionPool
from agent_platform.core.primitives.money import Money
from agent_platform.interface.commands import CommandError, ConsoleScope, MemoryCommand
from agent_platform.interface.commands.review import QueueCommand
from agent_platform.memory.working import WorkingMemoryStore
from customer_support_agent.console.review_handler import support_queue_command
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
    command = support_queue_command(pool)
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
        [table] = result.tables
        assert [row[1:3] for row in table.rows] == [["10001", "$20.00"]]
        assert table.headers[:3] == ["ID", "Order", "Amount"]

    async def test_another_customers_request_cannot_be_decided(
        self, queued: SqliteConnectionPool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        decided: list[Any] = []
        monkeypatch.setattr(
            "customer_support_agent.graph.builder.build_review_service",
            lambda pool=None, audit=None: decided.append(pool),
        )
        theirs = _id_for(queued, 2)
        with pytest.raises(CommandError, match="No pending review item"):
            await _queue(queued, _scope({"s": "1"}, ["s"])).run(
                ["approve", str(theirs), "looks right"]
            )
        assert decided == []

    async def test_nobody_verified_means_only_the_agents_own_items(
        self, queued: SqliteConnectionPool
    ) -> None:
        result = await _queue(queued, _scope({}, ["s"])).run([])
        assert result.ok and "0 pending" in result.title

    async def test_unscoped_it_is_the_whole_queue(self, queued: SqliteConnectionPool) -> None:
        result = await support_queue_command(queued).run([])
        assert "2 pending" in result.title

    async def test_a_decision_needs_a_comment(self, queued: SqliteConnectionPool) -> None:
        mine = _id_for(queued, 1)
        with pytest.raises(CommandError, match="comment"):
            await _queue(queued, _scope({"s": "1"}, ["s"])).run(["approve", str(mine)])
        assert QueueCommand(queued).service().queue.get(mine).status == "pending"  # type: ignore[union-attr]


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
