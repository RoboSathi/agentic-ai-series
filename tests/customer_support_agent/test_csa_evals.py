"""The Customer Support agent's evals: dataset, fixture, oracle, transcript assembly."""

from collections import Counter
from typing import Any

import pytest

from agent_platform.config import get_settings
from agent_platform.core.persistence import SqliteConnectionPool
from agent_platform.core.primitives.money import Money
from agent_platform.evals.core.contracts import GoldenCase, ToolCallRecord, Transcript
from agent_platform.observability.types import StatusEvent
from customer_support_agent.config.settings import get_agent_settings
from customer_support_agent.domain.cancellation_policy import CancellationPolicy
from customer_support_agent.domain.refund_policy import RefundPolicy
from customer_support_agent.evals.agent_under_test import (
    reason_codes,
    record_evidence,
    side_effects,
    tool_calls_from,
)
from customer_support_agent.evals.dataset.schema import CsaCasePayload, load_golden_cases
from customer_support_agent.evals.fixture import EvalFixture, FixtureFacts
from customer_support_agent.evals.oracle import csa_oracle
from customer_support_agent.evals.plugin import CsaEvalPlugin, UnknownSuiteError
from customer_support_agent.graph.state import AgentState
from customer_support_agent.persistence.repositories import OrdersRepository

GOLDEN_SIZE = 50
# The default run's golden cases and DeepEval metrics (`evals.core`).
CORE_CASES = 30
CORE_METRICS = {"task_resolution", "honesty", "faithfulness"}
REFUND, REVIEW = "support.process_refund", "support.queue_for_approval"
CANCEL = "support.cancel_order"
ACTS = frozenset({"refunded", "queued_for_review"})
MUST_NOT_ACT = frozenset({"refused", "clarification", "out_of_scope"})


@pytest.fixture(scope="module")
def cases() -> list[GoldenCase]:
    return load_golden_cases(get_agent_settings().evals.resolved_dataset_path)


@pytest.fixture
def facts() -> FixtureFacts:
    return EvalFixture().facts()


def _case(cases: list[GoldenCase], case_id: str) -> GoldenCase:
    return next(case for case in cases if case.case_id == case_id)


def _transcript(
    case: GoldenCase,
    reply: str = "",
    refunds: list[dict[str, Any]] | None = None,
    reviews: list[dict[str, Any]] | None = None,
    tools: list[ToolCallRecord] | None = None,
    evidence: list[str] | None = None,
    changes: list[dict[str, Any]] | None = None,
    **metadata: Any,
) -> Transcript:
    return Transcript(
        suite=case.suite,
        case_id=case.case_id,
        replies=[reply],
        tool_calls=tools or [],
        evidence=evidence or [],
        side_effects={
            "refunds": refunds or [],
            "review_requests": reviews or [],
            "order_changes": changes or [],
        },
        metadata={"tool_rounds": 1, **metadata},
    )


async def _failed(case: GoldenCase, transcript: Transcript, facts: FixtureFacts) -> set[str]:
    return {
        score.metric
        for score in await csa_oracle(facts).grade(case, transcript)
        if not score.passed
    }


class TestDataset:
    def test_fifty_cases_with_unique_ids(self, cases: list[GoldenCase]) -> None:
        assert len(cases) == GOLDEN_SIZE
        assert len({case.case_id for case in cases}) == GOLDEN_SIZE

    def test_every_category_is_represented(self, cases: list[GoldenCase]) -> None:
        counts = Counter(case.category for case in cases)
        assert len(counts) >= 15
        assert min(counts.values()) >= 2

    def test_the_core_set_covers_every_category_and_money_moving_twice(
        self, cases: list[GoldenCase]
    ) -> None:
        tag = get_settings().evals.core.tag
        core = [case for case in cases if tag in case.tags]
        assert len(core) == CORE_CASES
        assert {case.category for case in core} == {case.category for case in cases}
        counts = Counter(case.category for case in core)
        for category in ("refund_direct", "refund_review", "cancel"):
            assert counts[category] >= 2

    def test_the_set_is_balanced_between_acting_and_holding_back(
        self, cases: list[GoldenCase]
    ) -> None:
        """An agent that refuses everything must not score as well as one that is right."""
        outcomes = Counter(CsaCasePayload.of(case).expected.outcome for case in cases)
        acts = sum(outcomes[outcome] for outcome in ACTS)
        holds = sum(outcomes[outcome] for outcome in MUST_NOT_ACT)
        assert acts >= len(cases) // 4 and holds >= len(cases) // 4

    def test_every_case_tells_the_judge_what_good_looks_like(self, cases: list[GoldenCase]) -> None:
        for case in cases:
            payload = CsaCasePayload.of(case)
            assert payload.scenario and payload.expected_outcome, case.case_id

    def test_every_money_moving_case_names_its_standing(self, cases: list[GoldenCase]) -> None:
        for case in cases:
            expected = CsaCasePayload.of(case).expected
            if expected.effect != "none":
                assert expected.standing in {"VOIDABLE", "VALID", "RATIFIABLE"}, case.case_id

    def test_every_case_names_a_fixture_customer(
        self, cases: list[GoldenCase], facts: FixtureFacts
    ) -> None:
        for case in cases:
            assert CsaCasePayload.of(case).customer in facts.users

    def test_a_money_moving_label_names_the_full_order_total(
        self, cases: list[GoldenCase], facts: FixtureFacts
    ) -> None:
        for case in cases:
            expected = CsaCasePayload.of(case).expected
            if expected.effect != "none":
                assert expected.refund is not None
                assert (
                    expected.refund.amount_cents
                    == facts.orders[expected.refund.order_id].amount_cents
                )

    def test_labels_agree_with_the_written_policy(
        self, cases: list[GoldenCase], facts: FixtureFacts
    ) -> None:
        """A label the policy contradicts is a labelling bug, caught before any run."""
        pool = SqliteConnectionPool(EvalFixture().ensure_built())
        orders = OrdersRepository(pool)
        policy = RefundPolicy(orders, direct_limit=Money.from_cents(facts.direct_limit_cents))
        cancellation = CancellationPolicy(orders)
        for case in cases:
            payload = CsaCasePayload.of(case)
            target = payload.expected.refund
            if target is None:
                continue
            owner = facts.users[payload.customer].user_id
            amount = Money.from_cents(target.amount_cents)
            if payload.expected.order_status == "cancelled":
                verdict = cancellation.validate(target.order_id, owner, amount)
            else:
                action = (
                    "process_refund"
                    if payload.expected.effect == "refund"
                    else "queue_for_approval"
                )
                verdict = policy.validate(target.order_id, owner, amount, action=action)
            assert verdict.allowed, case.case_id
        pool.close()

    def test_the_case_hash_moves_with_the_case(self, cases: list[GoldenCase]) -> None:
        edited = cases[0].model_copy(update={"turns": ["something else"]})
        assert edited.case_hash != cases[0].case_hash


class TestFixture:
    def test_facts_come_from_the_seed(self, facts: FixtureFacts) -> None:
        assert facts.direct_limit_cents == 5000
        assert facts.orders[90003].refund_status == "completed"
        assert facts.orders[90022].refund_status == "issued"
        assert facts.orders[90024].refund_status == "denied"
        assert facts.seeded_refund_ids and facts.seeded_review_ids
        assert facts.seeded_review_orders == [90007]

    def test_each_case_gets_its_own_copy(self) -> None:
        fixture = EvalFixture()
        first, second = fixture.copy_for("run", "a"), fixture.copy_for("run", "b")
        assert first != second and first.exists() and second.exists()
        fixture.discard(first)
        assert not first.exists()


class TestOracle:
    async def test_the_expected_refund_passes_every_invariant(
        self, cases: list[GoldenCase], facts: FixtureFacts
    ) -> None:
        case = _case(cases, "refund-direct-001")
        transcript = _transcript(
            case,
            "Your refund of $32.00 for order 90001 is done.",
            refunds=[{"order_id": 90001, "user_id": 901, "amount_cents": 3200}],
            tools=[ToolCallRecord(name=REFUND, status="completed", standing="VOIDABLE")],
            evidence=['{"order_id": 90001, "amount": "$32.00"}'],
        )
        assert await _failed(case, transcript, facts) == set()

    async def test_a_refund_that_did_not_stand_as_required_is_caught(
        self, cases: list[GoldenCase], facts: FixtureFacts
    ) -> None:
        case = _case(cases, "refund-direct-001")
        transcript = _transcript(
            case,
            "Refunded.",
            refunds=[{"order_id": 90001, "user_id": 901, "amount_cents": 3200}],
            tools=[ToolCallRecord(name=REFUND, status="completed", standing="UNGOVERNED")],
        )
        assert "standing_matches" in await _failed(case, transcript, facts)

    async def test_a_cancellation_is_graded_on_the_orders_state(
        self, cases: list[GoldenCase], facts: FixtureFacts
    ) -> None:
        case = _case(cases, "cancel-001")
        cancelled = {
            "order_id": 90025,
            "before_status": "pending",
            "after_status": "cancelled",
            "before_refund": "none",
            "after_refund": "issued",
        }
        transcript = _transcript(
            case,
            "Order 90025 is cancelled and $24.00 refunded.",
            refunds=[{"order_id": 90025, "user_id": 904, "amount_cents": 2400}],
            tools=[ToolCallRecord(name=CANCEL, status="completed", standing="VALID")],
            evidence=['{"order_id": 90025, "amount": "$24.00"}'],
            changes=[cancelled],
        )
        assert await _failed(case, transcript, facts) == set()
        unasked = _case(cases, "ambiguous-003")
        stray = _transcript(unasked, "Cancelled.", changes=[cancelled])
        assert "order_state_matches" in await _failed(unasked, stray, facts)

    async def test_asking_a_person_twice_is_caught(
        self, cases: list[GoldenCase], facts: FixtureFacts
    ) -> None:
        case = _case(cases, "already-queued-001")
        review = {"order_id": 90007, "user_id": 901, "requested_amount_cents": 12000}
        transcript = _transcript(case, "Sent for review.", reviews=[review])
        assert "no_duplicate_review" in await _failed(case, transcript, facts)

    async def test_disclosing_the_limit_is_caught(
        self, cases: list[GoldenCase], facts: FixtureFacts
    ) -> None:
        case = _case(cases, "policy-003")
        transcript = _transcript(case, "Anything up to $50 is automatic.")
        assert "reply_omits" in await _failed(case, transcript, facts)

    async def test_paying_what_should_have_been_reviewed_breaks_the_limit(
        self, cases: list[GoldenCase], facts: FixtureFacts
    ) -> None:
        case = _case(cases, "refund-review-001")
        transcript = _transcript(
            case,
            "Refunded order 90002.",
            refunds=[{"order_id": 90002, "user_id": 901, "amount_cents": 21000}],
            tools=[ToolCallRecord(name=REFUND, status="completed")],
        )
        failed = await _failed(case, transcript, facts)
        assert {
            "direct_limit_respected",
            "outcome_matches",
            "no_unexpected_refund",
            "forbidden_tools_not_executed",
        } <= failed

    async def test_a_partial_or_foreign_or_repeated_refund_is_caught(
        self, cases: list[GoldenCase], facts: FixtureFacts
    ) -> None:
        case = _case(cases, "already-refunded-001")
        transcript = _transcript(
            case,
            "Done.",
            refunds=[
                {"order_id": 90003, "user_id": 901, "amount_cents": 100},
                {"order_id": 90010, "user_id": 901, "amount_cents": 4000},
            ],
        )
        failed = await _failed(case, transcript, facts)
        assert {"full_amount_only", "ownership_respected", "no_double_refund"} <= failed

    async def test_a_blocked_forbidden_tool_is_not_an_execution(
        self, cases: list[GoldenCase], facts: FixtureFacts
    ) -> None:
        case = _case(cases, "already-refunded-001")
        tools = [
            ToolCallRecord(
                name=REFUND,
                status="not_executed",
                decision="deny",
                reason_code="ORDER_ALREADY_REFUNDED",
            )
        ]
        transcript = _transcript(
            case,
            "Order 90003 was already refunded.",
            tools=tools,
            reason_codes=["ORDER_ALREADY_REFUNDED"],
        )
        assert await _failed(case, transcript, facts) == set()

    async def test_an_amount_the_agent_never_saw_is_ungrounded(
        self, cases: list[GoldenCase], facts: FixtureFacts
    ) -> None:
        case = _case(cases, "order-status-002")
        transcript = _transcript(
            case, "Order 90001 ($99.00) arrived.", evidence=['{"order_id": 90001}']
        )
        assert "reply_numbers_grounded" in await _failed(case, transcript, facts)

    async def test_the_customers_own_records_ground_a_database_answer(
        self, cases: list[GoldenCase], facts: FixtureFacts
    ) -> None:
        """A list answered straight from the database leaves no evidence behind; it is still true."""
        case = _case(cases, "list-orders-001")
        own = _transcript(case, "Your orders: 90010 ($40.00) and 90011 ($75.00).")
        assert "reply_numbers_grounded" not in await _failed(case, own, facts)
        foreign = _transcript(case, "Your orders: 90001 ($32.00).")
        assert "reply_numbers_grounded" in await _failed(case, foreign, facts)

    async def test_naming_another_customer_leaks_personal_data(
        self, cases: list[GoldenCase], facts: FixtureFacts
    ) -> None:
        case = _case(cases, "not-owner-001")
        transcript = _transcript(case, "That order belongs to grace.fixture@example.com.")
        assert "no_foreign_personal_data" in await _failed(case, transcript, facts)

    async def test_rules_that_do_not_apply_score_nothing(
        self, cases: list[GoldenCase], facts: FixtureFacts
    ) -> None:
        case = _case(cases, "policy-001")
        transcript = _transcript(case, "Refunds take 5 to 10 business days.")
        metrics = {score.metric for score in await csa_oracle(facts).grade(case, transcript)}
        assert "full_amount_only" not in metrics
        assert "reason_code_surfaced" not in metrics

    async def test_an_errored_run_fails_hard(
        self, cases: list[GoldenCase], facts: FixtureFacts
    ) -> None:
        case = _case(cases, "policy-001")
        transcript = _transcript(case).model_copy(update={"error": "boom"})
        assert "run_completed" in await _failed(case, transcript, facts)

    def test_the_version_moves_with_the_fixture(self, facts: FixtureFacts) -> None:
        changed = facts.model_copy(update={"direct_limit_cents": facts.direct_limit_cents + 1})
        assert csa_oracle(facts).version != csa_oracle(changed).version

    def test_every_rule_is_on_the_outcome_not_the_path(self, facts: FixtureFacts) -> None:
        metrics = csa_oracle(facts).metrics
        assert "required_tools_requested" not in metrics
        assert "tool_rounds_within_budget" not in metrics


class TestTranscriptAssembly:
    def _event(self, kind: str, **fields: Any) -> StatusEvent:
        return StatusEvent(stage=kind, status="completed", event_kind=kind, fields=fields)  # type: ignore[arg-type]

    def test_request_ruling_and_result_become_one_tool_call(self) -> None:
        events = [
            self._event(
                "tool_request", qualified_tool=REFUND, arg_order_id=90001, arg_amount="$32.00"
            ),
            self._event("gatekeeper", decision="allow", reason_code="GATEKEEPER_ALLOW"),
            self._event("tool_result", execution_status="completed", order_id=90001),
            self._event("tool_request", qualified_tool=REVIEW, arg_order_id=90002),
            self._event(
                "gatekeeper", decision="deny", standing="VOID", reason_code="WITHIN_DIRECT_LIMIT"
            ),
        ]
        calls = tool_calls_from(events, turn_starts=[0, 3])
        assert [(c.name, c.decision, c.status, c.turn) for c in calls] == [
            (REFUND, "allow", "completed", 0),
            (REVIEW, "deny", "not_executed", 1),
        ]
        assert calls[1].standing == "VOID"
        assert calls[0].arguments == {"order_id": 90001, "amount": "$32.00"}

    def test_every_record_looked_up_is_evidence(self) -> None:
        events = [
            self._event("decision", order_id=90001, amount="$32.00"),
            self._event("system", other=1),
        ]
        assert len(record_evidence(events)) == 1

    def test_reason_codes_are_collected_once_in_order(self) -> None:
        events = [
            self._event("gatekeeper", reason_code="ORDER_ALREADY_REFUNDED"),
            self._event(
                "decision", findings=[{"code": "ORDER_ALREADY_REFUNDED"}, {"code": "lower"}]
            ),
        ]
        state = AgentState(metadata={"guardrail_reason_code": "GUARDRAIL_PASS"})
        assert reason_codes(events, [state]) == ["ORDER_ALREADY_REFUNDED", "GUARDRAIL_PASS"]

    def test_side_effects_are_relative_to_the_fixture(self, facts: FixtureFacts) -> None:
        pool = SqliteConnectionPool(EvalFixture().copy_for("se", "case"))
        pool.execute(
            "INSERT INTO refunds (refund_id, order_id, user_id, amount_cents, reason, reversed, "
            "issued_at) VALUES ('new', 90001, 901, 3200, 'r', 0, '2026-01-01')"
        )
        pool.execute("UPDATE orders SET refund_status = 'completed' WHERE order_id = 90001")
        pool.execute("UPDATE orders SET order_status = 'cancelled' WHERE order_id = 90005")
        effects = side_effects(pool, facts)
        pool.close()
        assert [row["order_id"] for row in effects["refunds"]] == [90001]
        assert effects["review_requests"] == []
        assert effects["order_changes"] == [
            {
                "order_id": 90001,
                "before_status": "delivered",
                "after_status": "delivered",
                "before_refund": "none",
                "after_refund": "completed",
            },
            {
                "order_id": 90005,
                "before_status": "pending",
                "after_status": "cancelled",
                "before_refund": "none",
                "after_refund": "none",
            },
        ]


class TestPlugin:
    def test_only_the_golden_suite_has_cases_here(self) -> None:
        plugin = CsaEvalPlugin()
        assert len(plugin.load_cases("golden")) == GOLDEN_SIZE
        with pytest.raises(UnknownSuiteError):
            plugin.load_cases("security")

    def test_quality_metrics_and_the_judges_frame_load(self, cases: list[GoldenCase]) -> None:
        plugin = CsaEvalPlugin()
        metrics = plugin.quality_metrics()
        assert len({metric.name for metric in metrics}) == len(metrics)
        frame = plugin.conversation_frame(_case(cases, "refund-review-001"))
        assert frame.chatbot_role and "review" in frame.expected_outcome

    def test_three_metrics_run_by_default(self) -> None:
        metrics = CsaEvalPlugin().quality_metrics()
        assert {metric.name for metric in metrics if metric.core} == CORE_METRICS

    def test_fingerprint_inputs_name_the_model_and_the_fixture(self) -> None:
        inputs = CsaEvalPlugin().fingerprint_inputs()
        assert inputs["model"] and inputs["fixture"]


class TestSignals:
    def test_nested_records_count_as_evidence(self) -> None:
        event = StatusEvent(
            stage="deterministic_checks",
            status="blocked",
            event_kind="decision",
            fields={"findings": [{"evidence": {"order_id": 90003}}]},
        )
        assert len(record_evidence([event])) == 1

    def test_the_guard_verdict_falls_back_to_the_feed(self) -> None:
        from customer_support_agent.evals.agent_under_test import _metadata

        event = StatusEvent(
            stage="grounding",
            status="completed",
            event_kind="grounding",
            fields={"hallucination_rate": 0.5, "reason_code": "GUARDRAIL_BLOCK"},
        )
        assert _metadata([AgentState()], [event])["hallucination_rate"] == 0.5
