"""A Stop is not a failure: a cancelled call leaves the resilience stack as it found it.

The operator's Stop arrives as `CancelledError` mid-call. The breaker must not count it (a
stopped call says nothing about the dependency), retry must not try again (the operator asked
it to stop), and the bulkhead permit must come back (or the pool shrinks by one per Stop).
"""

import asyncio

import pytest

from agent_platform.resilience import ResilientCaller

_POOL = "stop-test"


async def _stopped_call(caller: ResilientCaller) -> None:
    started = asyncio.Event()

    async def held() -> str:
        started.set()
        await asyncio.Event().wait()
        return "never"

    task = asyncio.ensure_future(caller.call(held, idempotency_key="stop-test"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_the_breaker_does_not_count_a_stop() -> None:
    caller = ResilientCaller(_POOL)
    await _stopped_call(caller)
    assert caller.breaker.state == "closed" and caller.healthy


async def test_retry_does_not_try_again_after_a_stop() -> None:
    caller = ResilientCaller(_POOL)
    await _stopped_call(caller)
    assert caller.attempts == 1


async def test_the_bulkhead_permit_comes_back() -> None:
    caller = ResilientCaller(_POOL)
    size = caller.bulkhead.available(_POOL)  # type: ignore[attr-defined]
    await _stopped_call(caller)
    assert caller.bulkhead.available(_POOL) == size  # type: ignore[attr-defined]
