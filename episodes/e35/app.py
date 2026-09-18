"""E35 — Resilience Is a Control Plane Too.

Nygard's patterns, applied to an agent. The composition order is not
arbitrary:

    bulkhead → circuit breaker → retry → timeout → operation

Read outermost-in. Get it backwards and each layer undoes the one it
should protect: a timeout outside retry cuts off the retries instead of
the call, and a breaker inside retry counts one logical call as three
failures and trips at a third of the intended threshold.

The exit invariant: one saturated dependency must not starve another.
"""

import asyncio

from agentic_commons.resilience import (
    BulkheadFullError,
    CircuitBreakerImpl,
    CircuitBreakerOpenError,
    CompensatingSaga,
    ExponentialBackoffRetry,
    NonIdempotentRetryError,
    ResourcePoolBulkhead,
    SagaStep,
)
from episodes._shared import EpisodeOutput, main


async def run() -> EpisodeOutput:
    """Demonstrate each pattern, then the bulkhead isolation invariant."""
    output = EpisodeOutput(
        episode="E35",
        title="Resilience Is a Control Plane Too",
        principle="bulkhead → breaker → retry → timeout. Reverse it and each layer undoes the next.",
    )

    retry = ExponentialBackoffRetry(max_attempts=3, base_delay_seconds=0.001)
    try:
        await retry.execute(lambda: _ok(), "")
        keyed = False
    except NonIdempotentRetryError:
        keyed = True
    output.say("[bold]Retry requires an idempotency key[/bold] — enforced, not documented")
    output.say(f"  unkeyed retry refused: {keyed}")
    output.say("  A retry is a duplicate request by construction. Without a key,")
    output.say("  'the response was lost' and 'the customer was paid twice' are")
    output.say("  the same event from the caller's side.")

    breaker = CircuitBreakerImpl(name="dep", failure_threshold=2, half_open_after_seconds=0.05)
    for _ in range(2):
        try:
            await breaker.call(_boom)
        except RuntimeError:
            pass
    fast_failed = False
    try:
        await breaker.call(_ok)
    except CircuitBreakerOpenError:
        fast_failed = True
    output.say("")
    output.say(f"[bold]Circuit breaker[/bold]: {breaker.state}, failing fast: {fast_failed}")

    bulkhead = ResourcePoolBulkhead(pool_size=1)

    async def hold() -> None:
        async with bulkhead.acquire("slow"):
            await asyncio.sleep(0.15)

    task = asyncio.create_task(hold())
    await asyncio.sleep(0.02)
    starved = False
    try:
        async with bulkhead.acquire("slow", wait_seconds=0.02):
            pass
    except BulkheadFullError:
        starved = True
    async with bulkhead.acquire("fast", wait_seconds=0.02):
        unrelated_served = True
    await task

    output.say("")
    output.say("[bold]Bulkhead — the exit invariant[/bold]")
    output.say(f"  saturated pool refuses new work : {starved}")
    output.say(f"  an unrelated pool still serves  : {unrelated_served}")
    output.say("  One flooded compartment, not a flooded hull.")

    log: list[str] = []
    result = await CompensatingSaga().run(
        [
            _step(log, "mark_refunded"),
            _step(log, "record_payment"),
            _step(log, "notify", fail=True),
        ]
    )
    output.say("")
    output.say("[bold]Saga[/bold] — compensating in reverse")
    output.say(f"  failed at   : {result.failed_step}")
    output.say(f"  compensated : {result.compensated}")
    output.say(f"  order       : {log}")
    output.say("  Reverse, because a later step may depend on an earlier one.")

    return output.model_copy(
        update={
            "artifact": {
                "composition order": "bulkhead → breaker → retry → timeout → operation",
                "unkeyed retry refused": keyed,
                "breaker state": breaker.state,
                "isolation holds": starved and unrelated_served,
                "saga compensated": result.compensated,
            }
        }
    )


async def _ok() -> str:
    return "ok"


async def _boom() -> str:
    raise RuntimeError("down")


def _step(log: list[str], name: str, fail: bool = False) -> SagaStep:
    async def action() -> None:
        if fail:
            raise RuntimeError("crash")
        log.append(f"do:{name}")

    async def compensation() -> None:
        log.append(f"undo:{name}")

    return SagaStep(name=name, action=action, compensation=compensation)


if __name__ == "__main__":
    main(run)
