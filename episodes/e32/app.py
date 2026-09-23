"""E32 — PII & Sensitive Data: detection, redaction, minimization.

Three surfaces, and the third is the one people miss. PII in *input*
(what a tool call carries), PII in *memory* (what persists across
sessions), and PII in *logs* (what outlives the request).

Redacting only the visible response leaves the same data in three places
that survive it.

Redaction mode matters: masking is irreversible, tokenizing is stable so
records still join, removal destroys the shape of the text.
"""

from agentic_commons.security import PIIFilterImpl
from episodes._shared import EpisodeOutput, main

TEXT = (
    "Hi, I'm Ada — reach me at ada.lovelace@example.com or 555-123-4567. "
    "My card is 4111 1111 1111 1111 and my SSN is 123-45-6789."
)


async def run() -> EpisodeOutput:
    """Detect, redact three ways, and minimize a tool payload."""
    output = EpisodeOutput(
        episode="E32",
        title="PII: Detection, Redaction, Minimization",
        principle="Redacting the response leaves the same data in three places that outlive it.",
    )
    pii = PIIFilterImpl()

    spans = pii.detect(TEXT)
    output.say("[bold]Detection[/bold]")
    for span in spans:
        output.say(f"  {span.category:<13} {span.text}")
    output.say("  The card wins over the overlapping phone match — masking the")
    output.say("  partial one would leave the rest of the number in the clear.")

    output.say("")
    output.say("[bold]Three redaction modes, three guarantees[/bold]")
    for mode, note in [
        ("mask", "irreversible; the value is simply gone"),
        ("tokenize", "stable — the same value maps to the same token, so records still join"),
        ("remove", "destroys the shape of the text as well as the value"),
    ]:
        redacted = pii.redact(TEXT, mode)  # type: ignore[arg-type]
        output.say(f"  {mode:<9} {redacted[:66]}")
        output.say(f"            {note}")

    stable = pii.redact("ada.lovelace@example.com", "tokenize") == pii.redact(
        "ada.lovelace@example.com", "tokenize"
    )
    output.say(f"  tokenizing is deterministic: {stable}")

    payload = {"order_id": 55, "amount": 80.0, "ssn": "123-45-6789", "notes": "internal"}
    minimized = pii.minimize(payload)
    output.say("")
    output.say("[bold]Minimization[/bold] — an allow-list, not a deny-list")
    output.say(f"  before : {sorted(payload)}")
    output.say(f"  after  : {sorted(minimized)}")
    output.say("  A field added upstream is excluded until somebody decides it")
    output.say("  belongs, which is the failure direction you want.")

    return output.model_copy(
        update={
            "artifact": {
                "surfaces": "input (minimization), memory (provenance), logs (redaction before persistence)",
                "categories detected": [span.category for span in spans],
                "modes": "mask | tokenize | remove",
                "tokenizing stable": stable,
                "fields dropped": sorted(set(payload) - set(minimized)),
            }
        }
    )


if __name__ == "__main__":
    main(run)
