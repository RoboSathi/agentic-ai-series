"""Tracing, SLOs, cost and queue capacity (LLD §6)."""

import pytest

from agent_platform.observability import (
    CostAggregatorImpl,
    InMemoryLiveStatusFeed,
    LittlesLawQueueModel,
    LocalTracer,
    NullLiveStatusFeed,
    SLORegistry,
    Span,
    StageReporter,
    StatusEvent,
)
from agent_platform.observability.types import CostStack, utc_now


class TestStatusFeed:
    def test_subscribers_receive_events(self) -> None:
        feed = InMemoryLiveStatusFeed()
        seen: list[StatusEvent] = []
        feed.subscribe(seen.append)
        feed.emit(StatusEvent(stage="intake", status="started"))
        assert len(seen) == 1

    def test_a_failing_subscriber_cannot_break_a_request(self) -> None:
        """Emitting is cheap and never raises at the call site."""
        feed = InMemoryLiveStatusFeed()
        feed.subscribe(lambda event: (_ for _ in ()).throw(RuntimeError("ui broke")))
        feed.emit(StatusEvent(stage="intake", status="started"))
        assert len(feed.events) == 1

    def test_the_backlog_is_bounded(self) -> None:
        feed = InMemoryLiveStatusFeed(max_events=3)
        for index in range(10):
            feed.emit(StatusEvent(stage=str(index), status="started"))
        assert len(feed.events) == 3

    def test_events_are_filtered_by_run(self) -> None:
        feed = InMemoryLiveStatusFeed()
        feed.emit(StatusEvent(run_id="a", stage="x", status="started"))
        feed.emit(StatusEvent(run_id="b", stage="y", status="started"))
        assert len(list(feed.events_for("a"))) == 1

    def test_unsubscribing_stops_delivery(self) -> None:
        feed = InMemoryLiveStatusFeed()
        seen: list[StatusEvent] = []
        unsubscribe = feed.subscribe(seen.append)
        unsubscribe()
        feed.emit(StatusEvent(stage="x", status="started"))
        assert seen == []

    def test_the_null_feed_discards(self) -> None:
        feed = NullLiveStatusFeed()
        feed.emit(StatusEvent(stage="x", status="started"))
        assert list(feed.events_for("any")) == []

    def test_the_reporter_pairs_started_and_completed(self) -> None:
        feed = InMemoryLiveStatusFeed()
        reporter = StageReporter(feed, "run-1")
        reporter.started("llm_call")
        reporter.completed("llm_call", tokens_delta=120)
        assert [event.status for event in feed.events] == ["started", "completed"]
        assert feed.events[-1].tokens_delta == 120


class TestTracer:
    def test_a_span_is_recorded_with_its_authority_context(self) -> None:
        """A trace that cannot answer "why" is telemetry, not audit."""
        tracer = LocalTracer()
        with tracer.span("refund", actor="agent-1", tool="process_refund") as span:
            trace_id = span.trace_id
        recorded = tracer.spans_for(trace_id)
        assert recorded and recorded[0].actor == "agent-1"

    def test_a_span_records_its_latency(self) -> None:
        tracer = LocalTracer()
        with tracer.span("refund") as span:
            trace_id = span.trace_id
        assert tracer.spans_for(trace_id)[0].latency_ms >= 0

    def test_a_raising_body_still_closes_the_span(self) -> None:
        tracer = LocalTracer()
        with pytest.raises(RuntimeError):
            with tracer.span("refund"):
                raise RuntimeError("boom")
        assert tracer.spans

    def test_nested_spans_share_a_trace(self) -> None:
        tracer = LocalTracer()
        with tracer.span("outer") as outer:
            with tracer.span("inner", parent=outer) as inner:
                assert inner.parent_id == outer.span_id
                assert inner.trace_id == outer.trace_id


class TestSLOs:
    def test_records_and_averages_observations(self) -> None:
        registry = SLORegistry()
        registry.record("task_success_rate", 1.0)
        registry.record("task_success_rate", 0.0)
        assert registry.observed("task_success_rate") == 0.5

    def test_a_lower_is_better_metric_within_target_is_fine(self) -> None:
        registry = SLORegistry()
        registry.record("hallucination_rate", 0.01)
        assert registry.status("hallucination_rate").within_target is True

    def test_a_lower_is_better_metric_over_target_breaches(self) -> None:
        registry = SLORegistry()
        registry.record("hallucination_rate", 0.5)
        assert registry.status("hallucination_rate").within_target is False

    def test_a_zero_tolerance_metric_breaches_on_any_occurrence(self) -> None:
        """Unauthorized actions have a target of zero, not 'low'."""
        registry = SLORegistry()
        registry.record("unauthorized_action_rate", 0.01)
        assert registry.error_budget_remaining("unauthorized_action_rate") == 0.0

    def test_breaching_lists_only_the_failures(self) -> None:
        registry = SLORegistry()
        registry.record("hallucination_rate", 0.9)
        assert any(status.metric == "hallucination_rate" for status in registry.breaching())

    def test_an_unknown_metric_has_a_full_budget(self) -> None:
        assert SLORegistry().error_budget_remaining("invented") == 1.0


class TestCost:
    def test_accumulates_a_cost_stack_per_request(self) -> None:
        aggregator = CostAggregatorImpl()
        aggregator.record("r1", "input_tokens", 0.01)
        aggregator.record("r1", "output_tokens", 0.02)
        assert aggregator.stack_for("r1").total == pytest.approx(0.03)

    def test_a_span_attributes_its_own_cost(self) -> None:
        aggregator = CostAggregatorImpl()
        aggregator.observe_span(
            Span(trace_id="r1", span_id="s1", name="call", cost_usd=0.05, started_at=utc_now())
        )
        assert aggregator.total_cost == pytest.approx(0.05)

    def test_the_denominator_is_successes_only(self) -> None:
        """A cheaper but wrong answer is not a saving (E44)."""
        aggregator = CostAggregatorImpl()
        aggregator.record("r1", "output_tokens", 1.0)
        aggregator.record("r2", "output_tokens", 1.0)
        aggregator.record_outcome("r1", success=True)
        aggregator.record_outcome("r2", success=False)
        assert aggregator.cost_per_successful_outcome() == pytest.approx(2.0)

    def test_no_successes_reports_zero(self) -> None:
        assert CostAggregatorImpl().cost_per_successful_outcome() == 0.0

    def test_an_empty_stack_totals_zero(self) -> None:
        assert CostStack().total == 0.0


class TestQueueCapacity:
    def test_littles_law_worked_example(self) -> None:
        """E38: 100 cases/hr x 2 min => 3.33 reviewer-hours per hour."""
        model = LittlesLawQueueModel()
        assert model.required_capacity(100, 2) == pytest.approx(3.333, abs=0.01)

    def test_reports_whole_reviewers_needed(self) -> None:
        assert LittlesLawQueueModel().capacity_for(100, 2).reviewers_required == 4

    def test_an_understaffed_queue_is_unstable(self) -> None:
        """It does not get slower; it grows without bound."""
        assert LittlesLawQueueModel().is_stable(100, 2, reviewers=3) is False

    def test_an_adequately_staffed_queue_is_stable(self) -> None:
        assert LittlesLawQueueModel().is_stable(100, 2, reviewers=4) is True
