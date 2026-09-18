"""The refund agent's repositories, tools, saga and graph (LLD §21)."""

import pytest

from agentic_commons.persistence import SqliteConnectionPool
from customer_support_agent.core import (
    HitlQueueRepository,
    OrdersRepository,
    RefundLedgerRepository,
    RefundSaga,
    SupportTools,
    UsersRepository,
    build_agent,
    policy_text,
)
from tests.conftest import ScriptedLLM, tool_call


@pytest.fixture
def orders(seeded_pool: SqliteConnectionPool) -> OrdersRepository:
    return OrdersRepository(seeded_pool)


@pytest.fixture
def hitl(seeded_pool: SqliteConnectionPool) -> HitlQueueRepository:
    return HitlQueueRepository(seeded_pool)


@pytest.fixture
def refunds(seeded_pool: SqliteConnectionPool) -> RefundLedgerRepository:
    return RefundLedgerRepository(seeded_pool)


@pytest.fixture
def tools(
    orders: OrdersRepository, hitl: HitlQueueRepository, refunds: RefundLedgerRepository
) -> SupportTools:
    return SupportTools(orders, hitl, refunds, RefundSaga(orders, refunds))


class TestRepositories:
    def test_reads_a_seeded_order(self, orders: OrdersRepository) -> None:
        order = orders.get(10001)
        assert order is not None and order.amount == 32.00

    def test_a_missing_order_is_none(self, orders: OrdersRepository) -> None:
        assert orders.get(99999) is None

    def test_filters_by_user(self, orders: OrdersRepository) -> None:
        assert len(orders.for_user(1)) == 2

    def test_updates_a_refund_status(self, orders: OrdersRepository) -> None:
        assert orders.set_refund_status(10001, "completed") is True
        assert orders.get(10001).refund_status == "completed"  # type: ignore[union-attr]

    def test_updating_a_missing_order_reports_false(self, orders: OrdersRepository) -> None:
        assert orders.set_refund_status(99999, "completed") is False

    def test_already_refunded_is_derived_from_status(self, orders: OrdersRepository) -> None:
        assert orders.get(10003).already_refunded is True  # type: ignore[union-attr]

    def test_reads_a_user(self, seeded_pool: SqliteConnectionPool) -> None:
        assert UsersRepository(seeded_pool).get(1) is not None

    def test_reads_a_user_by_their_code(self, seeded_pool: SqliteConnectionPool) -> None:
        """The code is the credential; the id is the internal key."""
        from tests.conftest import CUSTOMER_CODE

        user = UsersRepository(seeded_pool).get_by_code(CUSTOMER_CODE)
        assert user is not None and user.user_id == 1

    def test_a_code_matches_regardless_of_case_or_spacing(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        """A customer copying from an email should not fail on capitals."""
        from tests.conftest import CUSTOMER_CODE

        assert UsersRepository(seeded_pool).get_by_code(f"  {CUSTOMER_CODE.lower()} ") is not None

    def test_an_unknown_code_matches_nothing(self, seeded_pool: SqliteConnectionPool) -> None:
        assert UsersRepository(seeded_pool).get_by_code("ZZZ9ZZZ") is None

    def test_queueing_and_deciding_a_hitl_request(
        self, hitl: HitlQueueRepository, tools: SupportTools
    ) -> None:
        tools.queue_for_approval(10002, 210.0)
        pending = hitl.pending()
        assert len(pending) == 1
        assert hitl.decide(pending[0].id, "approved") is True  # type: ignore[arg-type]

    def test_a_decided_request_cannot_be_decided_again(
        self, hitl: HitlQueueRepository, tools: SupportTools
    ) -> None:
        tools.queue_for_approval(10002, 210.0)
        request_id = hitl.pending()[0].id
        hitl.decide(request_id, "approved")  # type: ignore[arg-type]
        assert hitl.decide(request_id, "denied") is False  # type: ignore[arg-type]

    def test_the_refund_ledger_excludes_reversals(self, refunds: RefundLedgerRepository) -> None:
        refunds.record("r1", 10001, 1, 32.0, "damaged")
        refunds.record("r2", 10002, 1, 10.0, "goodwill")
        refunds.reverse("r2")
        assert refunds.total_for_user(1) == 32.0


class TestTools:
    def test_looking_up_an_order(self, tools: SupportTools) -> None:
        result = tools.get_order_status(10001)
        assert result.success and "32.00" in result.output

    def test_looking_up_a_missing_order_fails_cleanly(self, tools: SupportTools) -> None:
        assert tools.get_order_status(99999).success is False

    def test_refund_status_reports_a_pending_review(self, tools: SupportTools) -> None:
        tools.queue_for_approval(10002, 210.0)
        assert "pending" in tools.get_refund_status(10002).output

    async def test_a_refund_moves_the_order_to_completed(
        self, tools: SupportTools, orders: OrdersRepository
    ) -> None:
        result = await tools.process_refund(10001, 32.0, "damaged")
        assert result.success
        assert orders.get(10001).refund_status == "completed"  # type: ignore[union-attr]

    async def test_an_already_refunded_order_cannot_be_refunded_again(
        self, tools: SupportTools
    ) -> None:
        """Enforced in code, not in the prompt.

        Order 10003 belongs to u2, so the test authenticates as u2 — the
        default fixture binds u1, and to u1 that order does not exist.
        """
        from customer_support_agent.core.identity import Customer, set_current_customer

        set_current_customer(Customer(customer_id=2))
        result = await tools.process_refund(10003, 45.0)
        assert result.success is False and "already been refunded" in (result.error or "")

    async def test_refunding_a_missing_order_fails(self, tools: SupportTools) -> None:
        assert (await tools.process_refund(99999, 10.0)).success is False

    def test_queueing_marks_the_order_pending_review(
        self, tools: SupportTools, orders: OrdersRepository
    ) -> None:
        tools.queue_for_approval(10002, 210.0)
        assert orders.get(10002).refund_status == "pending_review"  # type: ignore[union-attr]

    def test_queueing_twice_does_not_duplicate_the_request(
        self, tools: SupportTools, hitl: HitlQueueRepository
    ) -> None:
        """A customer who asks twice must not produce two refunds."""
        tools.queue_for_approval(10002, 210.0)
        tools.queue_for_approval(10002, 210.0)
        assert len(hitl.pending()) == 1

    def test_goodwill_credit_is_recorded(
        self, tools: SupportTools, refunds: RefundLedgerRepository
    ) -> None:
        """The recipient is the authenticated customer, not an argument."""
        assert tools.apply_goodwill_credit(10.0, "sorry").success
        assert refunds.total_for_user(1) == 10.0

    def test_a_goodwill_credit_over_the_per_credit_cap_is_refused(
        self, tools: SupportTools
    ) -> None:
        """Money leaving the business is bounded, like a refund."""
        result = tools.apply_goodwill_credit(500.0, "sorry")
        assert result.success is False and "cannot exceed" in (result.error or "")

    def test_goodwill_credits_are_capped_per_customer(self, tools: SupportTools) -> None:
        """Otherwise the per-credit cap is bypassed by asking repeatedly."""
        for _ in range(4):
            tools.apply_goodwill_credit(25.0, "sorry")
        result = tools.apply_goodwill_credit(25.0, "sorry")
        assert result.success is False and "escalate instead" in (result.error or "")

    def test_a_negative_goodwill_credit_is_refused(self, tools: SupportTools) -> None:
        assert tools.apply_goodwill_credit(-50.0).success is False

    def test_every_tool_declares_its_governance_properties(self, tools: SupportTools) -> None:
        """A tool cannot be added without someone stating what it can break."""
        by_name = {tool.name: tool for tool in tools.definitions()}
        assert by_name["process_refund"].reversible is False
        assert by_name["queue_for_approval"].surface == "append_only"
        assert by_name["get_order_status"].blast_radius == "single_record"


class TestRefundSaga:
    async def test_a_successful_refund_completes_every_step(
        self, orders: OrdersRepository, refunds: RefundLedgerRepository
    ) -> None:
        saga = RefundSaga(orders, refunds)
        result = await saga.issue(orders.get(10001), 32.0, "damaged")  # type: ignore[arg-type]
        assert result.succeeded
        assert result.completed == ["mark_order_refunded", "record_payment", "notify_customer"]

    async def test_a_failure_rolls_the_order_status_back(
        self, orders: OrdersRepository, refunds: RefundLedgerRepository
    ) -> None:
        """Money moved with the order still unrefunded is the failure this prevents."""

        async def broken_notifier(user_id: str, message: str) -> None:
            raise RuntimeError("notification service down")

        saga = RefundSaga(orders, refunds, notifier=broken_notifier)
        order = orders.get(10001)
        assert order is not None
        result = await saga.issue(order, 32.0, "damaged")
        assert result.succeeded is False
        assert orders.get(10001).refund_status == "none"  # type: ignore[union-attr]

    async def test_a_failure_reverses_the_recorded_payment(
        self, orders: OrdersRepository, refunds: RefundLedgerRepository
    ) -> None:
        async def broken_notifier(user_id: str, message: str) -> None:
            raise RuntimeError("down")

        order = orders.get(10001)
        assert order is not None
        await RefundSaga(orders, refunds, notifier=broken_notifier).issue(order, 32.0, "x")
        assert refunds.total_for_user(1) == 0.0

    async def test_the_same_refund_is_not_issued_twice(
        self,
        orders: OrdersRepository,
        refunds: RefundLedgerRepository,
        seeded_pool: SqliteConnectionPool,
    ) -> None:
        from agentic_commons.resilience import SqliteIdempotencyGuard

        saga = RefundSaga(orders, refunds, idempotency=SqliteIdempotencyGuard(seeded_pool))
        order = orders.get(10001)
        assert order is not None
        await saga.issue(order, 32.0, "damaged")
        await saga.issue(order, 32.0, "damaged")
        assert refunds.total_for_user(1) == 32.0


class TestPolicy:
    def test_the_policy_loads_package_relative(self) -> None:
        """Not CWD-relative — it must work from any directory."""
        assert "Refund Policy" in policy_text()


class TestGraph:
    async def test_a_small_refund_is_processed(self, seeded_pool: SqliteConnectionPool) -> None:
        llm = ScriptedLLM(
            [
                {"tool_calls": tool_call("support.get_order_status", order_id=10001)},
                {"tool_calls": tool_call("support.process_refund", order_id=10001, amount=32.0)},
                {"content": "I've refunded $32.00."},
            ]
        )
        agent = build_agent(llm=llm, pool=seeded_pool, with_memory=False)
        state = await agent.handle("refund order 10001, it was damaged")
        assert OrdersRepository(seeded_pool).get(10001).refund_status == "completed"  # type: ignore[union-attr]
        assert state.response

    async def test_a_large_refund_is_refused_and_no_money_moves(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        """The model asks; the harness refuses. That is the whole design."""
        llm = ScriptedLLM(
            [
                {"tool_calls": tool_call("support.process_refund", order_id=10002, amount=210.0)},
                {"content": "That needs review."},
            ]
        )
        agent = build_agent(llm=llm, pool=seeded_pool, with_memory=False)
        state = await agent.handle("refund order 10002")
        assert state.escalated is True
        assert OrdersRepository(seeded_pool).get(10002).refund_status == "none"  # type: ignore[union-attr]

    async def test_an_answer_without_tools_is_returned(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        agent = build_agent(
            llm=ScriptedLLM([{"content": "Hello, how can I help?"}]),
            pool=seeded_pool,
            with_memory=False,
        )
        assert (await agent.handle("hello")).response == "Hello, how can I help?"

    async def test_the_tool_round_cap_still_produces_an_answer(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        """A cap is a control, not an excuse for an empty reply."""
        llm = ScriptedLLM([{"tool_calls": tool_call("support.get_order_status", order_id=10001)}])
        agent = build_agent(llm=llm, pool=seeded_pool, with_memory=False)
        state = await agent.handle("loop forever")
        assert state.tool_rounds <= 6

    async def test_an_invalid_tool_call_is_reported_to_the_model(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        llm = ScriptedLLM(
            [
                {"tool_calls": tool_call("support.process_refund")},
                {"content": "I need the order id."},
            ]
        )
        agent = build_agent(llm=llm, pool=seeded_pool, with_memory=False)
        state = await agent.handle("refund something")
        assert any("unavailable" in result for result in state.tool_results)

    async def test_the_graph_records_its_decisions(self, seeded_pool: SqliteConnectionPool) -> None:
        agent = build_agent(
            llm=ScriptedLLM([{"content": "hi"}]), pool=seeded_pool, with_memory=False
        )
        assert (await agent.handle("hello")).decisions

    async def test_the_policy_is_always_in_the_prompt(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        llm = ScriptedLLM([{"content": "hi"}])
        agent = build_agent(llm=llm, pool=seeded_pool, with_memory=False)
        await agent.handle("hello")
        system = llm.requests[0].messages[0].content
        assert "Refund Policy" in system
