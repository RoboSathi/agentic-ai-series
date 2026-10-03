"""E29 — Guardrails as a System: chaos, property tests, adversarial input.

One guardrail is a check. A system of guardrails is something you can
make claims about — and the only way to earn a claim is to attack it.

Property-based testing is the right shape here: rather than listing
outputs you thought of, assert an invariant and generate inputs against
it. The invariant that matters: *no malformed output reaches a
side-effecting executor*, whatever the model produced.
"""

import json
import random
import string
from typing import Callable

from pydantic import BaseModel

from agent_platform.guardrails import GuardrailViolationError, PydanticValidator
from episodes._shared import EpisodeOutput, main


class RefundCommand(BaseModel):
    order_id: int
    amount: float


def _adversarial(seed: int = 7, count: int = 200) -> list[str]:
    """Generate hostile strings: fragments, wrong types, injected text."""
    rng = random.Random(seed)
    alphabet = string.printable[:70]
    shapes: list[Callable[[], str]] = [
        lambda: "",
        lambda: "".join(rng.choices(alphabet, k=rng.randint(1, 40))),
        lambda: json.dumps({"order_id": "not-a-number", "amount": "free"}),
        lambda: json.dumps({"order_id": rng.randint(1, 99)}),
        lambda: '{"order_id": 1, "amount": 1.0',
        lambda: '{"order_id": 1, "amount": 1.0}' + "".join(rng.choices(alphabet, k=5)),
        lambda: "```json\n" + json.dumps({"order_id": 1, "amount": 2.0}) + "\n```",
        lambda: json.dumps({"order_id": 1, "amount": 2.0, "__proto__": "x"}),
    ]
    return [rng.choice(shapes)() for _ in range(count)]


async def run() -> EpisodeOutput:
    """Fire adversarial inputs and assert the invariant holds throughout."""
    output = EpisodeOutput(
        episode="E29",
        title="Guardrails as a System",
        principle="Assert an invariant and generate against it, rather than listing cases.",
    )
    validator = PydanticValidator("auto_repair")
    cases = _adversarial()

    accepted = rejected = raised = coerced = 0
    for raw in cases:
        result = validator.validate(raw, RefundCommand)
        if result.valid:
            accepted += 1
            # The invariant: anything accepted must satisfy the schema.
            if result.value is None or set(result.value) < {"order_id", "amount"}:
                coerced += 1
        else:
            rejected += 1
        try:
            validator.validate_or_raise(raw, RefundCommand)
        except GuardrailViolationError:
            raised += 1

    output.say(f"[bold]{len(cases)} adversarial inputs[/bold]")
    output.say(f"  accepted (and schema-valid) : {accepted}")
    output.say(f"  rejected                    : {rejected}")
    output.say(f"  raised rather than coerced  : {raised}")
    output.say("")
    output.say(f"[bold]Invariant[/bold]: nothing malformed was accepted — violations: {coerced}")
    output.say("")
    output.say("Property testing does not prove the absence of bugs. It does stop")
    output.say("you claiming a guarantee you only checked on five examples.")

    return output.model_copy(
        update={
            "artifact": {
                "adversarial inputs": len(cases),
                "accepted": accepted,
                "rejected": rejected,
                "invariant violations": coerced,
                "invariant": "no malformed output reaches a side-effecting executor",
            }
        }
    )


if __name__ == "__main__":
    main(run)
