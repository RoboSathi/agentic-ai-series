"""E28 — Hard Guarantees vs. Probabilistic Corrections.

A retry loop makes invalid output *less likely*. A grammar makes it
*impossible*. Those are different claims, and only one of them survives
contact with a tail.

The payoff is latency. A repair loop's cost is not its average, it is its
tail: the request that needed three attempts. Constrained decoding
removes the loop, so the tail disappears with it.

Honest note: of the configured backends only llama.cpp enforces a real
grammar. NIM accepts a hint and does not honour it, which is why the
harness validates the result either way.
"""

from pydantic import BaseModel

from agent_platform.config import get_settings
from agent_platform.guardrails import PydanticValidator
from episodes._shared import EpisodeOutput, main


class RefundCommand(BaseModel):
    order_id: int
    amount: float


# What a repair loop actually costs, per attempt, in round trips.
OUTPUTS = [
    '{"order_id": 1, "amount": 10.0}',
    '```json\n{"order_id": 2, "amount": 20.0}\n```',
    'Certainly: {"order_id": 3, "amount": 30.0,}',
    "sorry, I cannot",
]


async def run() -> EpisodeOutput:
    """Count attempts under repair, and contrast with a structural guarantee."""
    output = EpisodeOutput(
        episode="E28",
        title="Hard Guarantees vs. Probabilistic Corrections",
        principle="A retry makes it unlikely. A grammar makes it unrepresentable.",
    )
    validator = PydanticValidator("auto_repair")

    attempts: list[int] = []
    failures = 0
    output.say("[bold]Repair loop[/bold] — measured in attempts, not averages")
    for raw in OUTPUTS:
        result = validator.validate(raw, RefundCommand)
        attempts.append(result.attempts)
        failures += int(not result.valid)
        state = "valid" if result.valid else "UNFIXABLE"
        output.say(f"  {result.attempts} attempt(s)  {state:<10} {raw[:38]!r}")

    worst = max(attempts)
    output.say("")
    output.say(f"  mean attempts : {sum(attempts) / len(attempts):.2f}")
    output.say(f"  worst case    : {worst}")
    output.say(f"  unfixable     : {failures}")
    output.say("  The mean is reassuring. The tail is what pages someone.")

    backend = get_settings().guardrails.constrained_backend
    output.say("")
    output.say("[bold]Constrained decoding[/bold]")
    output.say(f"  configured backend : {backend}")
    output.say("  attempts           : 1, always — there is no invalid output to retry")
    output.say("")
    output.say("Of the backends here, only llama.cpp enforces a real grammar. NIM")
    output.say("accepts the hint and ignores it, so the result is still validated —")
    output.say("a best-effort hint is not a guarantee and must not be treated as one.")

    return output.model_copy(
        update={
            "artifact": {
                "repair mean attempts": sum(attempts) / len(attempts),
                "repair worst case": worst,
                "repair unfixable": failures,
                "constrained attempts": 1,
                "backend": backend,
                "caveat": "NIM does not honour guided_json; validation still runs",
            }
        }
    )


if __name__ == "__main__":
    main(run)
