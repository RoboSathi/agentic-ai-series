"""Shared sliding-window rate limiting for hosted endpoints (LLD §13)."""

import multiprocessing
from pathlib import Path

import httpx
import httpx2
import pytest

from agent_platform.config import get_settings
from agent_platform.config.settings import RateLimitBucketSettings, RateLimitSettings
from agent_platform.llm.http import create_http_client, create_sdk_http_client
from agent_platform.resilience import RateLimitWaitExceededError
from agent_platform.resilience.rate_limiter import (
    InMemoryRateWindowStore,
    RateLimitedHttpx2Transport,
    RateLimitedTransport,
    SlidingWindowRateLimiter,
    SqliteRateWindowStore,
    get_rate_limiter,
    reset_rate_limiter,
    retry_after_seconds,
)

LIMITED_HOST = "api.limited.test"
FREE_HOST = "localhost"
WINDOW = 60.0


def _settings(limit: int = 3, max_wait: float = 600.0) -> RateLimitSettings:
    return RateLimitSettings(
        enabled=True,
        store_path=":unused:",
        window_seconds=WINDOW,
        max_wait_seconds=max_wait,
        lock_timeout_seconds=5.0,
        default_retry_after_seconds=30.0,
        buckets={"hosted": RateLimitBucketSettings(max_requests=limit, hosts=[LIMITED_HOST])},
    )


class FakeClock:
    """Time that only moves when the limiter sleeps."""

    def __init__(self) -> None:
        self.now = 1_000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _limiter(limit: int = 3, max_wait: float = 600.0) -> tuple[SlidingWindowRateLimiter, FakeClock]:
    clock = FakeClock()
    limiter = SlidingWindowRateLimiter(
        _settings(limit, max_wait), InMemoryRateWindowStore(), clock=clock, sleep=clock.sleep
    )
    return limiter, clock


class TestSlidingWindow:
    async def test_admits_up_to_the_limit_without_waiting(self) -> None:
        limiter, clock = _limiter(limit=3)
        for _ in range(3):
            assert await limiter.acquire("hosted") == 0.0
        assert clock.sleeps == []

    async def test_the_next_request_waits_for_the_oldest_to_leave_the_window(self) -> None:
        limiter, clock = _limiter(limit=2)
        await limiter.acquire("hosted")
        clock.now += 10
        await limiter.acquire("hosted")
        waited = await limiter.acquire("hosted")
        assert waited == pytest.approx(WINDOW - 10)
        assert limiter.stats.bucket("hosted").throttled == 1

    async def test_no_window_ever_holds_more_than_the_limit(self) -> None:
        """The property that matters: a sliding window, not a fixed minute."""
        limiter, clock = _limiter(limit=5)
        admitted: list[float] = []
        for _ in range(40):
            await limiter.acquire("hosted")
            admitted.append(clock.now)
            clock.now += 1.5
        for start in admitted:
            in_window = [at for at in admitted if start <= at < start + WINDOW]
            assert len(in_window) <= 5

    async def test_a_wait_beyond_the_budget_fails_loudly(self) -> None:
        limiter, _ = _limiter(limit=1, max_wait=5.0)
        await limiter.acquire("hosted")
        with pytest.raises(RateLimitWaitExceededError):
            await limiter.acquire("hosted")

    async def test_a_429_blocks_the_bucket_for_retry_after(self) -> None:
        limiter, clock = _limiter(limit=10)
        limiter.penalize("hosted", 12.0)
        assert await limiter.acquire("hosted") == pytest.approx(12.0)
        assert limiter.stats.bucket("hosted").rejections == 1

    async def test_a_429_without_retry_after_uses_the_configured_default(self) -> None:
        limiter, _ = _limiter(limit=10)
        limiter.penalize("hosted", None)
        assert await limiter.acquire("hosted") == pytest.approx(30.0)

    def test_hosts_map_to_buckets_case_insensitively(self) -> None:
        limiter, _ = _limiter()
        assert limiter.bucket_for_host(LIMITED_HOST.upper()) == "hosted"
        assert limiter.bucket_for_host(FREE_HOST) is None


def _spend(path: str, attempts: int, results: "multiprocessing.Queue[float]") -> None:
    store = SqliteRateWindowStore(Path(path), lock_timeout_seconds=10.0)
    for _ in range(attempts):
        results.put(store.try_acquire("hosted", 7, WINDOW, 5_000.0))


class TestSharedStore:
    def test_the_sqlite_window_is_shared_by_separate_stores(self, tmp_path: Path) -> None:
        first = SqliteRateWindowStore(tmp_path / "rl.sqlite", 5.0)
        second = SqliteRateWindowStore(tmp_path / "rl.sqlite", 5.0)
        assert first.try_acquire("hosted", 1, WINDOW, 100.0) == 0.0
        assert second.try_acquire("hosted", 1, WINDOW, 101.0) == pytest.approx(WINDOW - 1)

    def test_concurrent_processes_never_overspend(self, tmp_path: Path) -> None:
        """Two processes racing for the last slots: exactly the limit is admitted."""
        path = str(tmp_path / "rl.sqlite")
        SqliteRateWindowStore(Path(path), 5.0)
        context = multiprocessing.get_context("spawn")
        results: "multiprocessing.Queue[float]" = context.Queue()
        workers = [context.Process(target=_spend, args=(path, 6, results)) for _ in range(2)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=60)
        waits = [results.get(timeout=5) for _ in range(12)]
        assert sum(1 for wait in waits if wait == 0.0) == 7

    def test_a_penalty_is_seen_by_every_store(self, tmp_path: Path) -> None:
        first = SqliteRateWindowStore(tmp_path / "rl.sqlite", 5.0)
        second = SqliteRateWindowStore(tmp_path / "rl.sqlite", 5.0)
        first.penalize("hosted", 200.0)
        assert second.try_acquire("hosted", 10, WINDOW, 150.0) == pytest.approx(50.0)


def _responder(status: int, headers: dict[str, str] | None = None):  # type: ignore[no-untyped-def]
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.host)
        return httpx.Response(status, headers=headers or {})

    return handler, seen


class TestTransport:
    async def test_every_attempt_to_a_limited_host_is_counted(self) -> None:
        limiter, _ = _limiter(limit=10)
        handler, _ = _responder(200)
        transport = RateLimitedTransport(httpx.MockTransport(handler), limiter)
        async with httpx.AsyncClient(transport=transport) as client:
            for _ in range(3):
                await client.get(f"https://{LIMITED_HOST}/v1/x")
        assert limiter.stats.bucket("hosted").requests == 3

    async def test_unlisted_hosts_pass_straight_through(self) -> None:
        limiter, _ = _limiter(limit=1)
        handler, _ = _responder(200)
        transport = RateLimitedTransport(httpx.MockTransport(handler), limiter)
        async with httpx.AsyncClient(transport=transport) as client:
            for _ in range(5):
                await client.get(f"http://{FREE_HOST}:8080/v1/x")
        assert limiter.stats.buckets == {}

    async def test_a_429_response_penalises_the_bucket(self) -> None:
        limiter, clock = _limiter(limit=10)
        handler, _ = _responder(429, {"Retry-After": "7"})
        transport = RateLimitedTransport(httpx.MockTransport(handler), limiter)
        async with httpx.AsyncClient(transport=transport) as client:
            await client.get(f"https://{LIMITED_HOST}/v1/x")
            await client.get(f"https://{LIMITED_HOST}/v1/x")
        assert clock.sleeps == [pytest.approx(7.0)]

    async def test_the_sdk_transport_applies_the_same_gate(self) -> None:
        limiter, _ = _limiter(limit=10)

        def handler(request: httpx2.Request) -> httpx2.Response:
            return httpx2.Response(200)

        transport = RateLimitedHttpx2Transport(httpx2.MockTransport(handler), limiter)
        async with httpx2.AsyncClient(transport=transport) as client:
            await client.get(f"https://{LIMITED_HOST}/v1/x")
        assert limiter.stats.bucket("hosted").requests == 1


class TestRetryAfter:
    def test_parses_delta_seconds(self) -> None:
        assert retry_after_seconds("4") == 4.0

    def test_parses_an_http_date_in_the_past_as_zero(self) -> None:
        assert retry_after_seconds("Wed, 21 Oct 2015 07:28:00 GMT") == 0.0

    def test_garbage_and_absence_mean_unknown(self) -> None:
        assert retry_after_seconds("soon") is None
        assert retry_after_seconds(None) is None


class TestFactory:
    def test_clients_use_the_limited_transport_when_enabled(self) -> None:
        reset_rate_limiter()
        assert get_rate_limiter() is not None
        client = create_http_client(1.0)
        assert isinstance(client._transport, RateLimitedTransport)
        sdk_client = create_sdk_http_client(1.0)
        assert isinstance(sdk_client._transport, RateLimitedHttpx2Transport)

    def test_the_bucket_limit_is_configuration(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from agent_platform.config import reload_settings

        monkeypatch.setenv("RESILIENCE__RATE_LIMIT__BUCKETS__NVIDIA__MAX_REQUESTS", "17")
        reload_settings()
        reset_rate_limiter()
        try:
            limiter = get_rate_limiter()
            assert limiter is not None
            assert limiter.limit_for("nvidia") == 17
            assert get_settings().resilience.rate_limit.buckets["nvidia"].hosts
        finally:
            monkeypatch.delenv("RESILIENCE__RATE_LIMIT__BUCKETS__NVIDIA__MAX_REQUESTS")
            reload_settings()
            reset_rate_limiter()

    def test_disabled_means_no_limiter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from agent_platform.config import reload_settings

        monkeypatch.setenv("RESILIENCE__RATE_LIMIT__ENABLED", "false")
        reload_settings()
        reset_rate_limiter()
        try:
            assert get_rate_limiter() is None
            assert not isinstance(create_http_client(1.0)._transport, RateLimitedTransport)
        finally:
            monkeypatch.delenv("RESILIENCE__RATE_LIMIT__ENABLED")
            reload_settings()
            reset_rate_limiter()
