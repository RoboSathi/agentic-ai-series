"""E45 — Build vs. Buy: choose the control surface, not the vendor.

The question is not which product is better. It is which controls you
need to own, because those are the ones you cannot delegate and still be
answerable for.

The mapping below is this repo's actual answer: what is built here, what
is bought, and — more usefully — *why* each one landed on that side.
"""

from episodes._shared import EpisodeOutput, main

DECISIONS = [
    (
        "Gatekeeper / authorization",
        "build",
        "the control you are answerable for; delegating it delegates the answer",
    ),
    ("Audit trail", "build", "must match your policy versions and survive a vendor change"),
    ("Memory write path", "build", "a security boundary, and the rules are yours"),
    ("Eval cascade", "build", "the scoring has to mean the same thing across every subsystem"),
    ("Model inference", "buy", "a commodity with a competitive market and a clean interface"),
    ("Embeddings", "buy", "same, and swapping is a config change"),
    (
        "Orchestration (LangGraph)",
        "buy",
        "state machine mechanics are well-solved; the invariant stays ours",
    ),
    (
        "Tracing backend (LangSmith)",
        "buy",
        "with a local tracer alongside, so the trail survives losing it",
    ),
    ("Terminal UI (Textual)", "buy", "rendering is not the differentiator"),
]


async def run() -> EpisodeOutput:
    """Print the build/buy decisions with their reasons."""
    output = EpisodeOutput(
        episode="E45",
        title="Build vs. Buy",
        principle="Own the controls you are answerable for. Buy the commodities.",
    )

    built = [row for row in DECISIONS if row[1] == "build"]
    bought = [row for row in DECISIONS if row[1] == "buy"]

    output.say("[bold]Built here[/bold]")
    for name, _, why in built:
        output.say(f"  {name:<30} {why}")

    output.say("")
    output.say("[bold]Bought[/bold]")
    for name, _, why in bought:
        output.say(f"  {name:<30} {why}")

    output.say("")
    output.say("[bold]The test[/bold]: if a regulator asks why an action was permitted,")
    output.say("can you answer without a vendor's help? Everything in the first")
    output.say("list is there because the answer had to be yes.")
    output.say("")
    output.say("Note what is bought *with a local equivalent alongside*: tracing.")
    output.say("Losing the vendor degrades the view, not the evidence chain.")

    return output.model_copy(
        update={
            "artifact": {
                "decisions": len(DECISIONS),
                "built": len(built),
                "bought": len(bought),
                "test": "can you answer a regulator without vendor help",
                "hedged": "tracing — LangSmith with LocalTracer alongside",
            }
        }
    )


if __name__ == "__main__":
    main(run)
