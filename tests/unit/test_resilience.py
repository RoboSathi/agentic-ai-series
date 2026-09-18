"""The seven Nygard patterns (LLD §13)."""

import asyncio

import pytest

from agentic_commons.persistence import SqliteConnectionPool
from agentic_commons.resilience import (
    AsyncTimeoutGuard,
    BulkheadFullError,
    CircuitBreakerImpl,
    CircuitBreakerOpenError,
    CompensatingSaga,
    Event,
    ExponentialBackoffRetry,
    InMemoryPubSub,
    NonIdempotentRetryError,
    ResilientCaller,
    ResourcePoolBulkhead,
    SagaStep,
    SqliteIdempotencyGuard,
    TimeoutOnlyCircuitBreaker,
)


async def _ok() -> str:
    return "ok"


async def _boom() -> str:
    raise RuntimeError("dependency down")


class TestCircuitBreaker:
    async def test_starts_closed(self) -> None:
        assert CircuitBreakerImpl().state == "closed"

    async def test_trips_open_at_the_threshold(self) -> None:
        breaker = CircuitBreakerImpl(failure_threshold=2)
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await breaker.call(_boom)
        assert breaker.state == "open"

    async def test_an_open_circuit_fails_fast(self) -> None:
        """The whole point: stop queueing behind a dependency known to be down."""
        breaker = CircuitBreakerImpl(failure_threshold=1)
        with pytest.raises(RuntimeError):
            await breaker.call(_boom)
        with pytest.raises(CircuitBreakerOpenError):
            await breaker.call(_ok)

    async def test_moves_to_half_open_after_the_cooldown(self) -> None:
        breaker = CircuitBreakerImpl(failure_threshold=1, half_open_after_seconds=0.05)
        with pytest.raises(RuntimeError):
            await breaker.call(_boom)
        await asyncio.sleep(0.06)
        assert breaker.state == "half_open"

    async def test_a_successful_trial_closes_the_circuit(self) -> None:
        breaker = CircuitBreakerImpl(failure_threshold=1, half_open_after_seconds=0.01)
        with pytest.raises(RuntimeError):
            await breaker.call(_boom)
        await asyncio.sleep(0.02)
        assert await breaker.call(_ok) == "ok"
        assert breaker.state == "closed"

    async def test_a_failed_trial_reopens_the_circuit(self) -> None:
        breaker = CircuitBreakerImpl(failure_threshold=1, half_open_after_seconds=0.01)
        with pytest.raises(RuntimeError):
            await breaker.call(_boom)
        await asyncio.sleep(0.02)
        with pytest.raises(RuntimeError):
            await breaker.call(_boom)
        assert breaker.state == "open"

    async def test_success_clears_the_failure_count(self) -> None:
        breaker = CircuitBreakerImpl(failure_threshold=3)
        with pytest.raises(RuntimeError):
            await breaker.call(_boom)
        await breaker.call(_ok)
        assert breaker.failure_count == 0

    def test_illegal_transitions_are_rejected(self) -> None:
        """The FSM table is enforced, not merely documented."""
        breaker = CircuitBreakerImpl()
        with pytest.raises(ValueError, match="Illegal circuit transition"):
            breaker._transition_to("closed")

    async def test_timeout_only_breaker_ignores_prompt_errors(self) -> None:
        """A dependency returning errors is answering — do not trip on it."""
        breaker = TimeoutOnlyCircuitBreaker(failure_threshold=1)
        with pytest.raises(RuntimeError):
            await breaker.call(_boom)
        assert breaker.state == "closed"

    async def test_timeout_only_breaker_trips_on_timeouts(self) -> None:
        async def slow() -> str:
            raise TimeoutError

        breaker = TimeoutOnlyCircuitBreaker(failure_threshold=1)
        with pytest.raises(TimeoutError):
            await breaker.call(slow)
        assert breaker.state == "open"


class TestRetry:
    async def test_refuses_to_retry_without_an_idempotency_key(self) -> None:
        """Enforced, not documented: this is how a customer is paid twice."""
        with pytest.raises(NonIdempotentRetryError):
            await ExponentialBackoffRetry().execute(_ok, "")

    async def test_retries_a_transient_failure(self) -> None:
        attempts = {"n": 0}

        async def flaky() -> str:
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise RuntimeError("transient")
            return "recovered"

        retry = ExponentialBackoffRetry(max_attempts=3, base_delay_seconds=0.001)
        assert await retry.execute(flaky, "key") == "recovered"

    async def test_gives_up_after_the_attempt_cap(self) -> None:
        retry = ExponentialBackoffRetry(max_attempts=2, base_delay_seconds=0.001)
        with pytest.raises(RuntimeError):
            await retry.execute(_boom, "key")

    async def test_does_not_retry_behind_an_open_breaker(self) -> None:
        """Retrying a known-down dependency only burns the budget."""
        attempts = {"n": 0}

        async def refused() -> str:
            attempts["n"] += 1
            raise CircuitBreakerOpenError("open")

        with pytest.raises(CircuitBreakerOpenError):
            await ExponentialBackoffRetry(max_attempts=5).execute(refused, "key")
        assert attempts["n"] == 1

    def test_backoff_is_bounded_by_the_max_delay(self) -> None:
        retry = ExponentialBackoffRetry(base_delay_seconds=1.0, max_delay_seconds=2.0)
        assert all(retry._delay_for(attempt) <= 2.0 for attempt in range(1, 10))


class TestTimeout:
    async def test_returns_a_fast_result(self) -> None:
        assert await AsyncTimeoutGuard().run(_ok, 1.0) == "ok"

    async def test_raises_past_the_deadline(self) -> None:
        async def slow() -> str:
            await asyncio.sleep(1)
            return "late"

        with pytest.raises(asyncio.TimeoutError):
            await AsyncTimeoutGuard().run(slow, 0.02)


class TestBulkhead:
    async def test_exhausting_one_pool_does_not_starve_another(self) -> None:
        """E35's exit invariant."""
        bulkhead = ResourcePoolBulkhead(pool_size=1)

        async def hold() -> None:
            async with bulkhead.acquire("slow"):
                await asyncio.sleep(0.2)

        task = asyncio.create_task(hold())
        await asyncio.sleep(0.02)
        with pytest.raises(BulkheadFullError):
            async with bulkhead.acquire("slow", wait_seconds=0.02):
                pass
        async with bulkhead.acquire("fast", wait_seconds=0.02):
            pass
        await task

    async def test_permits_are_released_on_exit(self) -> None:
        bulkhead = ResourcePoolBulkhead(pool_size=1)
        async with bulkhead.acquire("p"):
            assert bulkhead.in_flight("p") == 1
        assert bulkhead.in_flight("p") == 0

    async def test_permits_are_released_even_when_the_body_raises(self) -> None:
        bulkhead = ResourcePoolBulkhead(pool_size=1)
        with pytest.raises(RuntimeError):
            async with bulkhead.acquire("p"):
                raise RuntimeError("boom")
        assert bulkhead.available("p") == 1


class TestSaga:
    async def test_all_steps_succeed(self) -> None:
        log: list[str] = []
        steps = [_step(log, name) for name in ("a", "b")]
        result = await CompensatingSaga().run(steps)
        assert result.succeeded and result.completed == ["a", "b"]

    async def test_compensates_in_reverse_order(self) -> None:
        """Releasing a reservation before undoing what consumed it is wrong."""
        log: list[str] = []
        steps = [_step(log, "a"), _step(log, "b"), _step(log, "c", fail=True)]
        result = await CompensatingSaga().run(steps)
        assert result.succeeded is False
        assert result.failed_step == "c"
        assert log == ["do:a", "do:b", "undo:b", "undo:a"]

    async def test_a_failed_compensation_is_recorded_and_unwinding_continues(self) -> None:
        log: list[str] = []
        steps = [
            _step(log, "a"),
            _step(log, "b", compensation_fails=True),
            _step(log, "c", fail=True),
        ]
        result = await CompensatingSaga().run(steps)
        assert "b" in result.compensation_errors
        assert "a" in result.compensated

    async def test_a_step_without_compensation_is_skipped_on_unwind(self) -> None:
        log: list[str] = []
        steps = [
            SagaStep(name="a", action=_noop(log, "a"), compensation=None),
            _step(log, "b", fail=True),
        ]
        result = await CompensatingSaga().run(steps)
        assert result.compensated == []


class TestIdempotency:
    async def test_runs_once_per_key(self, pool: SqliteConnectionPool) -> None:
        guard = SqliteIdempotencyGuard(pool)
        runs = {"n": 0}

        async def work() -> dict[str, int]:
            runs["n"] += 1
            return {"paid": 1}

        first = await guard.once("k", work)
        second = await guard.once("k", work)
        assert first == second and runs["n"] == 1

    async def test_different_keys_run_separately(self, pool: SqliteConnectionPool) -> None:
        guard = SqliteIdempotencyGuard(pool)
        runs = {"n": 0}

        async def work() -> int:
            runs["n"] += 1
            return runs["n"]

        await guard.once("a", work)
        await guard.once("b", work)
        assert runs["n"] == 2

    def test_reports_whether_a_key_was_seen(self, pool: SqliteConnectionPool) -> None:
        guard = SqliteIdempotencyGuard(pool)
        assert guard.seen("k") is False
        guard.remember("k", {"v": 1})
        assert guard.seen("k") is True


class TestPubSub:
    async def test_delivers_to_every_subscriber(self) -> None:
        bus = InMemoryPubSub()
        seen: list[str] = []
        bus.subscribe("t", lambda event: seen.append("a"))
        bus.subscribe("t", lambda event: seen.append("b"))
        await bus.publish("t", Event(topic="t"))
        assert seen == ["a", "b"]

    async def test_a_failing_subscriber_does_not_stop_delivery(self) -> None:
        """A publisher must not learn that a subscriber broke."""
        bus = InMemoryPubSub()
        seen: list[str] = []
        bus.subscribe("t", lambda event: (_ for _ in ()).throw(RuntimeError("bad")))
        bus.subscribe("t", lambda event: seen.append("ok"))
        await bus.publish("t", Event(topic="t"))
        assert seen == ["ok"]

    async def test_cancelling_a_subscription_stops_delivery(self) -> None:
        bus = InMemoryPubSub()
        seen: list[str] = []
        subscription = bus.subscribe("t", lambda event: seen.append("x"))
        subscription.cancel()
        await bus.publish("t", Event(topic="t"))
        assert seen == [] and subscription.active is False

    async def test_supports_async_handlers(self) -> None:
        bus = InMemoryPubSub()
        seen: list[str] = []

        async def handler(event: Event) -> None:
            seen.append("async")

        bus.subscribe("t", handler)
        await bus.publish("t", Event(topic="t"))
        assert seen == ["async"]


class TestComposition:
    async def test_the_stack_passes_a_successful_call_through(self) -> None:
        caller = ResilientCaller("dep")
        assert await caller.call(_ok, idempotency_key="k") == "ok"

    async def test_retries_happen_inside_the_breaker(self) -> None:
        """Three retries of one logical call must count as one breaker failure."""
        caller = ResilientCaller(
            "dep",
            breaker=CircuitBreakerImpl(name="dep", failure_threshold=2),
            retry=ExponentialBackoffRetry(max_attempts=3, base_delay_seconds=0.001),
        )
        with pytest.raises(RuntimeError):
            await caller.call(_boom, idempotency_key="k")
        assert caller.breaker.state == "closed"


def _noop(log: list[str], name: str):  # type: ignore[no-untyped-def]
    async def action() -> None:
        log.append(f"do:{name}")

    return action


def _step(
    log: list[str], name: str, fail: bool = False, compensation_fails: bool = False
) -> SagaStep:
    async def action() -> None:
        if fail:
            raise RuntimeError("crash")
        log.append(f"do:{name}")

    async def compensation() -> None:
        if compensation_fails:
            raise RuntimeError("compensation failed")
        log.append(f"undo:{name}")

    return SagaStep(name=name, action=action, compensation=compensation)
