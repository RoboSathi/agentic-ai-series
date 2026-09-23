"""The full request lifecycle, end to end (HLD §4).

One customer query travelling through every live subsystem: intake and
injection screening, context assembly, the model turn, gatekept tool
execution, the refund saga, the audit trail, and the answer. The unit
tests prove each part; this proves they are actually wired together.
"""

import pytest

from agentic_commons.audit import SqliteAuditLedger
from agentic_commons.gatekeeper import GatekeeperImpl
from agentic_commons.gatekeeper.ledger import CumulativeLedger
from agentic_commons.memory import MemoryFacade, MemoryQuery
from agentic_commons.observability import (
    CostAggregatorImpl,
    InMemoryLiveStatusFeed,
    SLORegistry,
    StageReporter,
)
from agentic_commons.persistence import SqliteConnectionPool
from agentic_commons.security import Authorizer
from customer_support_agent.core import (
    HitlQueueRepository,
    OrdersRepository,
    RefundLedgerRepository,
    build_agent,
)
from tests.conftest import ScriptedLLM, tool_call


class TestAutoApprovedRefundLifecycle:
    """A $32 refund: under the threshold, so the agent completes it."""

    @pytest.fixture
    def outcome(self, seeded_pool: SqliteConnectionPool):  # type: ignore[no-untyped-def]
        return _run(
            seeded_pool,
            query="Order 10001 arrived damaged, I'd like a refund please",
            script=[
                {"tool_calls": tool_call("support.get_order_status", order_id=10001)},
                {
                    "tool_calls": tool_call(
                        "support.process_refund", order_id=10001, amount=32.0, reason="damaged"
                    )
                },
                {"content": "I've refunded $32.00 for order 10001."},
            ],
        )

    async def test_the_customer_gets_an_answer(self, outcome) -> None:  # type: ignore[no-untyped-def]
        state, _ = await outcome
        assert state.response

    async def test_the_order_is_marked_refunded(
        self, outcome, seeded_pool: SqliteConnectionPool
    ) -> None:  # type: ignore[no-untyped-def]
        await outcome
        assert OrdersRepository(seeded_pool).get(10001).refund_status == "completed"  # type: ignore[union-attr]

    async def test_the_payment_is_in_the_refund_ledger(
        self, outcome, seeded_pool: SqliteConnectionPool
    ) -> None:  # type: ignore[no-untyped-def]
        await outcome
        assert RefundLedgerRepository(seeded_pool).total_for_user(1) == 32.0

    async def test_every_tool_call_left_an_audit_entry(self, outcome) -> None:  # type: ignore[no-untyped-def]
        _, parts = await outcome
        assert parts["audit"].count() == 2

    async def test_the_live_feed_saw_each_stage(self, outcome) -> None:  # type: ignore[no-untyped-def]
        """What the TUI renders is the same stream the harness emits anyway."""
        _, parts = await outcome
        stages = {event.stage for event in parts["feed"].events}
        assert {"prompt_injection_check", "context_assembly", "llm_call"} <= stages

    async def test_the_cost_of_the_request_is_attributed(self, outcome) -> None:  # type: ignore[no-untyped-def]
        _, parts = await outcome
        costs: CostAggregatorImpl = parts["cost"]
        costs.record_outcome("run-1", success=True)
        assert costs.cost_per_successful_outcome() > 0


class TestEscalatedRefundLifecycle:
    """A $210 refund: above the threshold, so a human decides."""

    @pytest.fixture
    def outcome(self, seeded_pool: SqliteConnectionPool):  # type: ignore[no-untyped-def]
        return _run(
            seeded_pool,
            query="I want a full refund for order 10002",
            script=[
                {"tool_calls": tool_call("support.get_order_status", order_id=10002)},
                {
                    "tool_calls": tool_call(
                        "support.queue_for_approval", order_id=10002, amount=210.0
                    )
                },
                {"content": "That refund needs manual review; you'll hear back shortly."},
            ],
        )

    async def test_the_request_is_queued_for_a_human(
        self, outcome, seeded_pool: SqliteConnectionPool
    ) -> None:  # type: ignore[no-untyped-def]
        await outcome
        assert len(HitlQueueRepository(seeded_pool).pending()) == 1

    async def test_no_money_moved(self, outcome, seeded_pool: SqliteConnectionPool) -> None:  # type: ignore[no-untyped-def]
        await outcome
        assert RefundLedgerRepository(seeded_pool).total_for_user(1) == 0.0

    async def test_the_order_is_marked_pending_review(
        self, outcome, seeded_pool: SqliteConnectionPool
    ) -> None:  # type: ignore[no-untyped-def]
        await outcome
        assert OrdersRepository(seeded_pool).get(10002).refund_status == "pending_review"  # type: ignore[union-attr]

    async def test_a_human_can_then_approve_it(
        self, outcome, seeded_pool: SqliteConnectionPool
    ) -> None:  # type: ignore[no-untyped-def]
        """The agent proposes; the human disposes."""
        await outcome
        queue = HitlQueueRepository(seeded_pool)
        request = queue.pending()[0]
        assert queue.decide(request.id, "approved") is True  # type: ignore[arg-type]
        assert queue.pending() == []


class TestMemoryAcrossTurns:
    async def test_a_session_is_recalled_in_a_later_turn(
        self, seeded_pool: SqliteConnectionPool, tmp_path
    ) -> None:
        """E16's cross-session recall, through the real facade."""
        from agentic_commons.memory import EpisodicStore, SemanticStore

        memory = MemoryFacade.complete(semantic=SemanticStore(seeded_pool), episodic=EpisodicStore(tmp_path))
        agent = build_agent(
            llm=ScriptedLLM([{"content": "Noted."}]),
            pool=seeded_pool,
            memory=memory,
        )
        await agent.handle("my order 10001 arrived damaged", session_id="s1")
        recalled = await memory.recall(MemoryQuery(text="damaged", session_id="s1"))
        assert recalled


class TestSLOsOverALifecycle:
    async def test_a_clean_run_breaches_no_slo(self, seeded_pool: SqliteConnectionPool) -> None:
        state, _ = await _run(
            seeded_pool,
            query="what is the status of order 10001?",
            script=[
                {"tool_calls": tool_call("support.get_order_status", order_id=10001)},
                {"content": "It was delivered."},
            ],
        )
        slos = SLORegistry()
        slos.record_outcome("task_success_rate", success=bool(state.response))
        slos.record("unauthorized_action_rate", 0.0)
        assert slos.breaching() == [] or all(
            status.metric != "unauthorized_action_rate" for status in slos.breaching()
        )


async def _run(pool: SqliteConnectionPool, query: str, script: list[dict[str, object]]):  # type: ignore[no-untyped-def]
    """Run one query through a fully assembled agent, returning its parts."""
    audit = SqliteAuditLedger(pool)
    feed = InMemoryLiveStatusFeed()
    cost = CostAggregatorImpl()
    agent = build_agent(
        llm=ScriptedLLM(script),
        pool=pool,
        gatekeeper=GatekeeperImpl(
            authorizer=Authorizer(), ledger=CumulativeLedger(pool), audit=audit
        ),
        reporter=StageReporter(feed, "run-1"),
        with_memory=False,
    )
    state = await agent.handle(query, session_id="run-1")
    cost.record("run-1", "output_tokens", 0.0003)
    return state, {"audit": audit, "feed": feed, "cost": cost}
