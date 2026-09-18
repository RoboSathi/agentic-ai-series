"""E23 — State, Events & Durable Execution.

An agent run is a state machine whether or not anybody wrote one down.
Writing it down means illegal transitions are rejected rather than
silently tolerated, and a crash resumes from a checkpoint rather than
from the beginning.

The canonical worked example is the circuit breaker: three states, four
legal transitions, and an explicit table. Every other transition raises.
"""

from agentic_commons.control_loop import Checkpoint, Transition
from agentic_commons.resilience import CircuitBreakerImpl, CircuitBreakerOpenError
from episodes._shared import EpisodeOutput, main

LEGAL = {
    "IDLE": {"PLANNING"},
    "PLANNING": {"ACTING", "FAILED"},
    "ACTING": {"WAITING_TOOL", "WAITING_HUMAN", "DONE", "FAILED"},
    "WAITING_TOOL": {"ACTING", "FAILED"},
    "WAITING_HUMAN": {"ACTING", "FAILED"},
    "DONE": set(),
    "FAILED": set(),
}


async def run() -> EpisodeOutput:
    """Show the agent FSM, then the breaker enforcing its own table."""
    output = EpisodeOutput(
        episode="E23",
        title="State, Events & Durable Execution",
        principle="An illegal transition should raise, not be tolerated.",
    )

    output.say("[bold]The agent's legal transitions[/bold]")
    for state, targets in LEGAL.items():
        allowed = ", ".join(sorted(targets)) or "(terminal)"
        output.say(f"  {state:<14} → {allowed}")

    breaker = CircuitBreakerImpl(name="payments", failure_threshold=2, half_open_after_seconds=0.05)

    async def failing() -> str:
        raise RuntimeError("payment gateway down")

    async def working() -> str:
        return "ok"

    output.say("")
    output.say("[bold]The circuit breaker — the same discipline, applied to a dependency[/bold]")
    output.say(f"  start                : {breaker.state}")
    for _ in range(2):
        try:
            await breaker.call(failing)
        except RuntimeError:
            pass
    output.say(f"  after 2 failures     : {breaker.state}")
    try:
        await breaker.call(working)
    except CircuitBreakerOpenError:
        output.say("  next call            : refused immediately, without touching the dependency")

    import asyncio

    await asyncio.sleep(0.06)
    output.say(f"  after the cooldown   : {breaker.state}  (one trial call allowed)")
    await breaker.call(working)
    output.say(f"  trial succeeded      : {breaker.state}")

    try:
        breaker._transition_to("open")
        breaker._transition_to("closed")
        illegal_rejected = False
    except ValueError as exc:
        illegal_rejected = True
        output.say(f"  illegal transition   : rejected — {exc}")

    checkpoint = Checkpoint(
        run_id="run-1",
        state="WAITING_HUMAN",
        transition_history=[
            Transition(from_state="IDLE", to_state="PLANNING"),
            Transition(from_state="PLANNING", to_state="ACTING"),
            Transition(from_state="ACTING", to_state="WAITING_HUMAN"),
        ],
        durable_facts={"order_id": 55, "amount": 210.0},
    )
    output.say("")
    output.say(
        f"[bold]A checkpoint[/bold]: resume at {checkpoint.state} with {len(checkpoint.durable_facts)} durable facts"
    )
    output.say(f"  history: {' → '.join(t.to_state for t in checkpoint.transition_history)}")

    return output.model_copy(
        update={
            "artifact": {
                "agent states": len(LEGAL),
                "breaker states": "closed → open → half_open",
                "illegal transition rejected": illegal_rejected,
                "checkpoint state": checkpoint.state,
                "durable facts": len(checkpoint.durable_facts),
            }
        }
    )


if __name__ == "__main__":
    main(run)
