"""The refund agent's repositories, tools, saga and graph (LLD §21)."""

from types import SimpleNamespace
from typing import Any

import pytest

from agent_platform.config import get_settings, set_enabled
from agent_platform.core.persistence import SqliteConnectionPool
from agent_platform.core.primitives.money import Money
from customer_support_agent.graph import build_agent
from customer_support_agent.persistence import (
    HitlQueueRepository,
    OrdersRepository,
    RefundLedgerRepository,
    UsersRepository,
)
from customer_support_agent.prompts.policy import policy_text
from customer_support_agent.resilience import RefundSaga
from customer_support_agent.tools import SupportTools
from tests.conftest import ScriptedLLM, refunded_to, tool_call


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
        assert order is not None and order.amount == Money.parse("32.00")

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

    def test_the_refund_ledger_excludes_reversals(
        self, refunds: RefundLedgerRepository, seeded_pool: SqliteConnectionPool
    ) -> None:
        refunds.record("r1", 10001, 1, Money.parse("32.00"), "damaged")
        refunds.record("r2", 10002, 1, Money.parse("10.00"), "refund")
        refunds.reverse("r2")
        assert refunded_to(seeded_pool, 1) == Money.parse("32.00")


class TestTools:
    def test_looking_up_an_order(self, tools: SupportTools) -> None:
        result = tools.get_order_status(10001)
        assert result.success and "32.00" in result.output

    def test_looking_up_a_missing_order_fails_cleanly(self, tools: SupportTools) -> None:
        assert tools.get_order_status(99999).success is False

    def test_refund_status_reports_a_pending_review(self, tools: SupportTools) -> None:
        tools.queue_for_approval(10002, 210.0)
        assert "pending" in tools.get_refund_status(10002).output

    async def test_a_refund_moves_the_order_to_issued(
        self, tools: SupportTools, orders: OrdersRepository
    ) -> None:
        """Issued, not completed: it settles later, and until then it can be voided."""
        result = await tools.process_refund(10001, 32.0, "damaged")
        assert result.success
        order = orders.get(10001)
        assert order is not None
        assert order.refund_status == "issued"
        assert order.refund_settles_at is not None

    async def test_an_already_refunded_order_cannot_be_refunded_again(
        self, tools: SupportTools
    ) -> None:
        """Enforced in code, not in the prompt.

        Order 10003 belongs to u2, so the test authenticates as u2 — the
        default fixture binds u1, and to u1 that order does not exist.
        """
        from customer_support_agent.domain.identity import Customer, set_current_customer

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


class _BrokenLedger(RefundLedgerRepository):
    """A ledger whose write fails: the payment step of the saga cannot commit."""

    def record(self, *args: object, **kwargs: object) -> None:
        raise RuntimeError("ledger unavailable")


class TestRefundSaga:
    async def test_a_successful_refund_completes_every_step(
        self, orders: OrdersRepository, refunds: RefundLedgerRepository
    ) -> None:
        saga = RefundSaga(orders, refunds)
        result = await saga.issue(orders.get(10001), 32.0, "damaged")  # type: ignore[arg-type]
        assert result.succeeded
        assert result.completed == ["mark_refund_issued", "record_payment"]

    async def test_a_failure_rolls_the_order_status_back(
        self, orders: OrdersRepository, seeded_pool: SqliteConnectionPool
    ) -> None:
        """An order marked refunded with no payment recorded is the failure this prevents."""
        order = orders.get(10001)
        assert order is not None
        result = await RefundSaga(orders, _BrokenLedger(seeded_pool)).issue(order, 32.0, "x")
        assert result.succeeded is False
        restored = orders.get(10001)
        assert restored is not None
        assert restored.refund_status == "none"
        assert restored.refund_settles_at is None

    async def test_a_stale_order_is_not_refunded(
        self,
        orders: OrdersRepository,
        refunds: RefundLedgerRepository,
        seeded_pool: SqliteConnectionPool,
    ) -> None:
        """The issue is one conditional write: a snapshot that no longer holds pays nothing."""
        stale = orders.get(10001)
        assert stale is not None
        orders.set_refund_status(10001, "completed")
        result = await RefundSaga(orders, refunds).issue(stale, 32.0, "x")
        assert result.succeeded is False
        assert refunded_to(seeded_pool, 1) == Money.zero()

    async def test_a_partial_amount_is_not_issued(
        self, orders: OrdersRepository, refunds: RefundLedgerRepository
    ) -> None:
        """The full total is part of the conditional write, not only of the tool's check."""
        order = orders.get(10001)
        assert order is not None
        result = await RefundSaga(orders, refunds).issue(order, 1.0, "x")
        assert result.succeeded is False
        assert orders.get(10001).refund_status == "none"  # type: ignore[union-attr]

    async def test_the_same_refund_is_not_issued_twice(
        self,
        orders: OrdersRepository,
        refunds: RefundLedgerRepository,
        seeded_pool: SqliteConnectionPool,
    ) -> None:
        from agent_platform.resilience import SqliteIdempotencyGuard

        saga = RefundSaga(orders, refunds, idempotency=SqliteIdempotencyGuard(seeded_pool))
        order = orders.get(10001)
        assert order is not None
        await saga.issue(order, 32.0, "damaged")
        await saga.issue(order, 32.0, "damaged")
        assert refunded_to(seeded_pool, 1) == Money.parse("32.00")


class TestPolicy:
    def test_the_policy_loads_package_relative(self) -> None:
        """Not CWD-relative — it must work from any directory."""
        assert "Refund Policy" in policy_text()


class TestGraph:
    async def test_the_tool_round_cap_still_produces_an_answer(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        """A cap is a control, not an excuse for an empty reply."""
        llm = ScriptedLLM([{"tool_calls": tool_call("support.get_order_status", order_id=10001)}])
        agent = build_agent(llm=llm, pool=seeded_pool, with_memory=False)
        state = await agent.handle("loop forever")
        assert state.tool_rounds <= 6

    async def test_the_graph_records_its_decisions(self, seeded_pool: SqliteConnectionPool) -> None:
        agent = build_agent(
            llm=ScriptedLLM([{"content": "hi"}]), pool=seeded_pool, with_memory=False
        )
        assert (await agent.handle("hello")).decisions


class TestTheBuilderHonoursConfiguration:
    """Which subsystems are wired is a config decision, not a literal.

    This is what makes an episode possible: the same `build_agent()` the
    capstone runs, under a narrower config, produces a narrower agent.
    Before this, two hardcoded booleans decided it and everything else
    was wired unconditionally.

    Leaving a collaborator out is deliberately not the same as letting it
    disable itself internally — a component that is still constructed is
    still called and still narrates a step to anyone watching the trace.
    """

    def _build(self, seeded_pool: SqliteConnectionPool, off: str = "") -> object:
        """Build from an everything-on posture, minus one capability.

        Starting from all-on rather than from the declared config is
        deliberate: whoever is working here may have switched a feature
        off for their own demo, and this class is testing the wiring
        mechanism, not their current posture.

        Features declared in the `features:` block are flipped there;
        anything still carrying a plain section `enabled:` flag goes
        through the older toggle. Both routes end at the same question —
        what did the composition root wire?
        """
        settings = get_settings()
        features = settings.features.with_all(True)
        if off and off in features.paths():
            features = features.with_override(off, False)
        elif off:
            set_enabled(off, None, False)
        return build_agent(llm=ScriptedLLM(), pool=seeded_pool, features=features)

    def _off(self, section: str, seeded_pool: SqliteConnectionPool) -> object:
        """Build an agent with one capability switched off."""
        return self._build(seeded_pool, off=section)

    def test_grounding_does_not_depend_on_observability(
        self, seeded_pool: SqliteConnectionPool, restore_settings: None
    ) -> None:
        """The defect this guards, which was real.

        Grounding used to be gated on the `observe` flag alongside the
        tracer and the SLO registry. "Is this reply supported by what the
        tools returned?" is a hallucination-protection question, so an
        agent configured to be quiet was also an agent that had quietly
        stopped checking its own answers.
        """
        agent = self._off("observability", seeded_pool)
        assert agent._tracer is None
        assert agent._grounding is not None


class TestAWithheldFeatureLeavesNoStage:
    """The claim an episode makes on screen, asserted.

    Two halves, and the second is what makes it a demonstration rather
    than a breakage: the stage is gone from the compiled graph, *and* the
    turn still runs to an answer. An agent that crashes without its
    grounding check has not shown you what grounding was doing.
    """

    STAGES = [
        ("deterministic_checks", "deterministic_checks"),
        ("prompt_injection", "prompt_injection"),
        ("content_moderation", "content_moderation"),
        ("intent_classification", "classify_intent"),
        ("context", "assemble_context"),
    ]

    def _agent(self, feature: str, on: bool, pool: SqliteConnectionPool) -> object:
        settings = get_settings()
        return build_agent(
            llm=ScriptedLLM([{"content": "Order 10001 was delivered."}]),
            pool=pool,
            features=settings.features.with_override(feature, on),
        )

    def _nodes(self, agent: object) -> set[str]:
        return set(agent._graph.get_graph().nodes)  # type: ignore[attr-defined]

    @pytest.mark.parametrize(("feature", "node"), STAGES)
    def test_the_stage_is_present_when_declared(
        self, feature: str, node: str, seeded_pool: SqliteConnectionPool, restore_settings: None
    ) -> None:
        assert node in self._nodes(self._agent(feature, True, seeded_pool))

    @pytest.mark.parametrize(("feature", "node"), STAGES)
    def test_the_stage_is_absent_when_withheld(
        self, feature: str, node: str, seeded_pool: SqliteConnectionPool, restore_settings: None
    ) -> None:
        assert node not in self._nodes(self._agent(feature, False, seeded_pool))

    @pytest.mark.parametrize(("feature", "node"), STAGES)
    async def test_the_turn_still_completes_without_it(
        self, feature: str, node: str, seeded_pool: SqliteConnectionPool, restore_settings: None
    ) -> None:
        agent = self._agent(feature, False, seeded_pool)
        state = await agent.handle("what is the status of order 10001?")  # type: ignore[attr-defined]
        assert state.response

    async def test_every_stage_withheld_at_once_still_answers(
        self, seeded_pool: SqliteConnectionPool, restore_settings: None
    ) -> None:
        """`/harness off` — the E01 gesture — must not produce a broken agent."""
        settings = get_settings()
        agent = build_agent(
            llm=ScriptedLLM([{"content": "Order 10001 was delivered."}]),
            pool=seeded_pool,
            features=settings.features.with_all(False),
        )
        nodes = self._nodes(agent)
        assert not {node for _, node in self.STAGES} & nodes
        assert (await agent.handle("what is the status of order 10001?")).response


class TestAWithheldStageSaysNothing:
    """Reported from a live run: the guard was off and the trace still showed it.

    `hallucination_guard` is not a graph node — it is a step inside
    `finalize` — so withholding it left `ResponseGuardrail` in place with
    no detector, returning the draft unexamined while `finalize` printed
    "Grounded: 0/0 claims supported" regardless. The screen was telling an
    operator a control was running when it was not, which is worse than
    showing nothing at all.
    """

    class _Feed:
        def __init__(self) -> None:
            self.events: list[object] = []

        def publish(self, event: object) -> None:
            self.events.append(event)

        def emit(self, event: object) -> None:
            self.events.append(event)

    async def _stages(self, on: bool, pool: SqliteConnectionPool) -> list[str]:
        from agent_platform.observability import StageReporter

        feed = self._Feed()
        agent = build_agent(
            llm=ScriptedLLM([{"content": "Order 10001 was delivered on March 3."}]),
            pool=pool,
            features=get_settings().features.with_override("hallucination_guard", on),
            reporter=StageReporter(feed, "test"),
        )
        await agent.handle("please help me with something about my account", session_id="s")
        return [str(getattr(event, "stage", "")) for event in feed.events]

    async def test_it_says_nothing_at_all_when_it_was_not(
        self, seeded_pool: SqliteConnectionPool, restore_settings: None
    ) -> None:
        assert "hallucination_check" not in await self._stages(False, seeded_pool)


def _turn_state(metadata: dict[str, Any]) -> Any:
    """The parts of a turn `_offered_tools` reads: its metadata, and its query.

    The query matters now: the turn's skills narrow what is offered (ADR 0010), and
    they are selected from it. An empty one selects none, so nothing extra is narrowed.
    """
    return SimpleNamespace(metadata=metadata, query="", model_query="")


class TestHarnessOffLeavesTheModelFree:
    """Reported from a live run: with the harness off, no tool ever ran.

    Switching the harness off is meant to produce an agent free to act
    with nothing standing between its proposals and execution. It
    produced the opposite. `_offered_tools` reads `intent_tools` from
    state, and with `intent_classification` withheld nothing writes that
    key — so an *absent* classifier was read as a classifier that
    permitted nothing, and the model was handed an empty toolset. The
    demonstration inverted itself: the unguarded agent was the one that
    could do nothing at all.
    """

    def _agent(self, on: bool, pool: SqliteConnectionPool) -> Any:
        features = get_settings().features.with_all(True)
        return build_agent(
            llm=ScriptedLLM(
                [
                    {"tool_calls": tool_call("support.get_order_status", order_id=10001)},
                    {"content": "Order 10001 was delivered on March 3."},
                ]
            ),
            pool=pool,
            features=features.with_override("intent_classification", on),
        )

    async def test_without_a_classifier_the_model_is_still_offered_tools(
        self, seeded_pool: SqliteConnectionPool, restore_settings: None
    ) -> None:
        agent = self._agent(False, seeded_pool)
        offered = await agent._offered_tools(_turn_state({}))
        assert offered, "an absent classifier narrowed nothing, so nothing is excluded"

    async def test_a_classifier_that_permits_nothing_still_offers_nothing(
        self, seeded_pool: SqliteConnectionPool, restore_settings: None
    ) -> None:
        """The distinction the fix turns on, asserted from the other side."""
        agent = self._agent(True, seeded_pool)
        offered = await agent._offered_tools(_turn_state({"intent": "UNKNOWN", "intent_tools": []}))
        assert offered == []

    async def test_the_security_narrowing_still_applies_without_a_classifier(
        self, seeded_pool: SqliteConnectionPool, restore_settings: None
    ) -> None:
        """Losing intent routing must not lose the injection guard's scoping."""
        agent = self._agent(False, seeded_pool)
        offered = await agent._offered_tools(
            _turn_state({"allowed_actions": ["support.get_order_status"]})
        )
        assert [tool.qualified_name for tool in offered] == ["support.get_order_status"]


class TestAnUnplacedTurnIsNeverOfferedAWrite:
    """No skill active means "no narrowing", which must not become "every tool"."""

    async def test_an_unplaced_turn_is_offered_reads_only(
        self, seeded_pool: SqliteConnectionPool, restore_settings: None
    ) -> None:
        agent = TestHarnessOffLeavesTheModelFree()._agent(True, seeded_pool)
        state = _turn_state(
            {"intent": "UNKNOWN", "intent_tools": [], "intent_execution_route": "llm_fallback"}
        )
        names = {tool.name for tool in await agent._offered_tools(state)}
        assert names == {"get_order_status", "get_refund_status", "list_orders"}


class TestAnUnplacedTurnMayBeHintedAReadSkill:
    @staticmethod
    def _intent(*scores: tuple[str, float]) -> Any:
        from customer_support_agent.graph.nodes.intent import IntentNodes

        candidates = [SimpleNamespace(intent=name, score=score) for name, score in scores]
        intent = SimpleNamespace(name="UNKNOWN", candidates=candidates)
        return IntentNodes._suggested_intent(intent)

    def test_a_clear_read_candidate_below_the_threshold_is_the_hint(self) -> None:
        assert self._intent(("UNKNOWN", 0.4), ("ORDER_STATUS", 0.5), ("LIST_ORDERS", 0.1)) == (
            "ORDER_STATUS"
        )

    def test_a_write_candidate_is_never_the_hint(self) -> None:
        assert self._intent(("UNKNOWN", 0.3), ("REQUEST_REFUND", 0.55)) is None

    def test_two_close_candidates_give_no_hint(self) -> None:
        assert self._intent(("ORDER_STATUS", 0.45), ("ORDER_DETAILS", 0.42)) is None

    def test_a_weak_candidate_gives_no_hint(self) -> None:
        assert self._intent(("UNKNOWN", 0.8), ("ORDER_STATUS", 0.1)) is None
