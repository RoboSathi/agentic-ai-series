"""The Customer Support agent's evals: dataset, fixture, oracle, transcript assembly."""

from collections import Counter
from typing import Any

import pytest

from agent_platform.evals.contracts import GoldenCase, ToolCallRecord, Transcript
from agent_platform.observability.types import StatusEvent
from agent_platform.persistence import SqliteConnectionPool
from agent_platform.primitives.money import Money
from customer_support_agent.config.settings import get_agent_settings
from customer_support_agent.domain.refund_policy import RefundPolicy
from customer_support_agent.evals.agent_under_test import (
    reason_codes,
    record_evidence,
    side_effects,
    tool_calls_from,
)
from customer_support_agent.evals.dataset.schema import CsaCasePayload, load_golden_cases
from customer_support_agent.evals.fixture import EvalFixture, FixtureFacts
from customer_support_agent.evals.oracle import CsaOracle
from customer_support_agent.evals.plugin import CsaEvalPlugin, UnknownSuiteError
from customer_support_agent.graph.state import AgentState
from customer_support_agent.persistence.repositories import OrdersRepository

GOLDEN_SIZE = 50
REFUND, REVIEW = "support.process_refund", "support.queue_for_approval"


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
            "order_changes": [],
        },
        metadata={"tool_rounds": 1, **metadata},
    )


async def _failed(case: GoldenCase, transcript: Transcript, facts: FixtureFacts) -> set[str]:
    return {
        score.metric for score in await CsaOracle(facts).grade(case, transcript) if not score.passed
    }


class TestDataset:
    def test_fifty_cases_with_unique_ids(self, cases: list[GoldenCase]) -> None:
        assert len(cases) == GOLDEN_SIZE
        assert len({case.case_id for case in cases}) == GOLDEN_SIZE

    def test_every_category_is_represented(self, cases: list[GoldenCase]) -> None:
        counts = Counter(case.category for case in cases)
        assert len(counts) == 12
        assert min(counts.values()) >= 2

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
        policy = RefundPolicy(
            OrdersRepository(pool), direct_limit=Money.from_cents(facts.direct_limit_cents)
        )
        for case in cases:
            payload = CsaCasePayload.of(case)
            target = payload.expected.refund
            if target is None:
                continue
            owner = facts.users[payload.customer].user_id
            amount = Money.from_cents(target.amount_cents)
            action = (
                "process_refund" if payload.expected.effect == "refund" else "queue_for_approval"
            )
            assert policy.validate(target.order_id, owner, amount, action=action).allowed, (
                case.case_id
            )
        pool.close()

    def test_the_case_hash_moves_with_the_case(self, cases: list[GoldenCase]) -> None:
        edited = cases[0].model_copy(update={"turns": ["something else"]})
        assert edited.case_hash != cases[0].case_hash


class TestFixture:
    def test_facts_come_from_the_seed(self, facts: FixtureFacts) -> None:
        assert facts.direct_limit_cents == 5000
        assert facts.orders[90003].refund_status == "completed"
        assert facts.seeded_refund_ids and facts.seeded_review_ids

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
            tools=[ToolCallRecord(name=REFUND, status="completed")],
            evidence=['{"order_id": 90001, "amount": "$32.00"}'],
        )
        assert await _failed(case, transcript, facts) == set()

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
        case = _case(cases, "order-status-004")
        transcript = _transcript(
            case, "Order 90001 ($99.00) arrived.", evidence=['{"order_id": 90001}']
        )
        assert "reply_numbers_grounded" in await _failed(case, transcript, facts)

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
        metrics = {score.metric for score in await CsaOracle(facts).grade(case, transcript)}
        assert "full_amount_only" not in metrics
        assert "reason_code_surfaced" not in metrics

    async def test_an_errored_run_fails_hard(
        self, cases: list[GoldenCase], facts: FixtureFacts
    ) -> None:
        case = _case(cases, "policy-001")
        transcript = _transcript(case).model_copy(update={"error": "boom"})
        assert "run_completed" in await _failed(case, transcript, facts)

    def test_the_version_moves_with_the_rules(self, facts: FixtureFacts) -> None:
        full, fewer = CsaOracle(facts), CsaOracle(facts, invariants=())
        assert full.version != fewer.version


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
            self._event("gatekeeper", decision="deny", reason_code="WITHIN_DIRECT_LIMIT"),
        ]
        calls = tool_calls_from(events)
        assert [(c.name, c.decision, c.status) for c in calls] == [
            (REFUND, "allow", "completed"),
            (REVIEW, "deny", "not_executed"),
        ]
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
        effects = side_effects(pool, facts)
        pool.close()
        assert [row["order_id"] for row in effects["refunds"]] == [90001]
        assert effects["review_requests"] == []
        assert effects["order_changes"] == [
            {"order_id": 90001, "before": "none", "after": "completed"}
        ]


class TestPlugin:
    def test_only_the_golden_suite_has_cases_here(self) -> None:
        plugin = CsaEvalPlugin()
        assert len(plugin.load_cases("golden")) == GOLDEN_SIZE
        with pytest.raises(UnknownSuiteError):
            plugin.load_cases("security")

    def test_rubric_and_quality_metrics_load(self) -> None:
        plugin = CsaEvalPlugin()
        assert plugin.judge_rubric("golden").criteria
        assert {metric.kind for metric in plugin.quality_metrics()} <= {
            "answer_relevancy",
            "faithfulness",
            "geval",
        }

    def test_fingerprint_inputs_name_the_model(self) -> None:
        assert CsaEvalPlugin().fingerprint_inputs()["model"]


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
