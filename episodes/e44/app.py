"""E44 — Cost & Performance Engineering: Amdahl, caching, cascading.

The denominator is the whole argument. Cost per *call* rewards a cheaper
model that gets the answer wrong; cost per *successful outcome* does not,
because a wrong answer contributes spend and no success.

Model cascading is the NOLLM cascade applied to money: cheapest
sufficient tier first, escalating only when the cheap tier's answer will
not do.
"""

from agent_platform.observability import CostAggregatorImpl
from episodes._shared import EpisodeOutput, main

# (request, tier, cost, succeeded)
TRAFFIC = [
    ("r1", "cheap", 0.0004, True),
    ("r2", "cheap", 0.0004, True),
    ("r3", "cheap", 0.0004, False),
    ("r4", "expensive", 0.0090, True),
    ("r5", "cheap", 0.0004, False),
    ("r6", "expensive", 0.0090, True),
]


async def run() -> EpisodeOutput:
    """Contrast cost per call with cost per successful outcome."""
    output = EpisodeOutput(
        episode="E44",
        title="Cost & Performance Engineering",
        principle="A cheaper wrong answer is not a saving.",
    )
    costs = CostAggregatorImpl()

    for request, tier, cost, succeeded in TRAFFIC:
        costs.record(request, "output_tokens", cost)
        costs.record_outcome(request, success=succeeded)

    successes = sum(1 for *_, ok in TRAFFIC if ok)
    per_call = costs.total_cost / len(TRAFFIC)
    per_success = costs.cost_per_successful_outcome()

    output.say(f"[bold]{len(TRAFFIC)} requests, {successes} successful[/bold]")
    for request, tier, cost, succeeded in TRAFFIC:
        output.say(f"  {request}  {tier:<10} ${cost:.4f}  {'ok' if succeeded else 'FAILED'}")

    output.say("")
    output.say(f"  total spend              : ${costs.total_cost:.4f}")
    output.say(f"  cost per call            : ${per_call:.4f}")
    output.say(f"  cost per success (E44)   : ${per_success:.4f}")
    output.say("")
    output.say("The gap is the failures. Optimising the first number rewards")
    output.say("routing more traffic to a tier that gets it wrong.")

    stack = costs.stack_for("r1")
    output.say("")
    output.say("[bold]The cost stack[/bold] — spend has components, not one number")
    output.say("  input tokens, output tokens, tool calls, retrieval,")
    output.say("  evaluator calls, human review, orchestration, infrastructure")
    output.say("  human review is in there: escalating is a cost line, not a free safety net")

    return output.model_copy(
        update={
            "artifact": {
                "requests": len(TRAFFIC),
                "successes": successes,
                "total spend": costs.total_cost,
                "cost per call": per_call,
                "cost per successful outcome": per_success,
                "penalty for failure": f"{(per_success / per_call - 1):.0%} higher",
                "stack components": len(stack.model_dump()),
            }
        }
    )


if __name__ == "__main__":
    main(run)
