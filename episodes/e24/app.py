"""E24 — Structured Outputs, Parsing & Fail-Loud Interfaces.

Three modes, and they are not the same guarantee.

`reject` and `auto_repair` validate after generation: the model can still
produce anything, you find out afterwards, and a repair costs a round
trip. `constrained` constrains the decode itself, so invalid output is
not unlikely but unrepresentable.

The failure this prevents is silent coercion — an agent acting on a value
no model ever produced.
"""

from pydantic import BaseModel

from agent_platform.guardrails import GuardrailViolationError, PydanticValidator
from episodes._shared import EpisodeOutput, main


class RefundCommand(BaseModel):
    order_id: int
    amount: float


MALFORMED = [
    ("well-formed", '{"order_id": 55, "amount": 80.0}'),
    ("wrapped in a code fence", '```json\n{"order_id": 55, "amount": 80.0}\n```'),
    ("prefaced with prose", 'Sure! Here you go: {"order_id": 55, "amount": 80.0}'),
    ("trailing comma", '{"order_id": 55, "amount": 80.0,}'),
    ("not JSON at all", "I cannot do that."),
]


async def run() -> EpisodeOutput:
    """Run each malformation through reject and auto_repair."""
    output = EpisodeOutput(
        episode="E24",
        title="Structured Outputs & Fail-Loud Interfaces",
        principle="Validation is what you do when you cannot constrain the decoder.",
    )
    strict = PydanticValidator("reject")
    repairing = PydanticValidator("auto_repair")

    output.say("[bold]reject[/bold] — halt on anything invalid")
    for label, raw in MALFORMED:
        result = strict.validate(raw, RefundCommand)
        output.say(f"  {label:<24} {'accepted' if result.valid else 'rejected'}")

    output.say("")
    output.say("[bold]auto_repair[/bold] — bounded, mechanical fixes only")
    repaired = 0
    for label, raw in MALFORMED:
        result = repairing.validate(raw, RefundCommand)
        state = "accepted" if result.valid else "rejected"
        note = f" (repaired in {result.attempts} attempts)" if result.repaired else ""
        repaired += int(result.repaired)
        output.say(f"  {label:<24} {state}{note}")

    output.say("")
    output.say("The repairs are deliberately mechanical — strip a fence, cut to the")
    output.say("outermost braces, drop a trailing comma. Anything cleverer is")
    output.say("guessing what the model meant, which is the failure being prevented.")

    try:
        strict.validate_or_raise("I cannot do that.", RefundCommand)
        raised = False
    except GuardrailViolationError as exc:
        raised = True
        output.say("")
        output.say(f"[bold]fail loud[/bold]: {exc}")

    return output.model_copy(
        update={
            "artifact": {
                "modes": "reject | auto_repair | constrained",
                "cases": len(MALFORMED),
                "repaired under auto_repair": repaired,
                "raised rather than coerced": raised,
                "constrained backend": "grammar (a real grammar); a hosted catalogue honours only a hint",
            }
        }
    )


if __name__ == "__main__":
    main(run)
