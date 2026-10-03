"""E10 — NOLLM First: the eval cascade and the cost of measurement.

Six tiers, each cheaper and more deterministic than the one below it.
Reaching the LLM judge is a failure to resolve cheaply, not a first move.

The headline number is `nollm_scorer_ratio`: the share of judgements
settled without a model. It is measurable precisely because there is one
cascade, shared by evals, retrieval, tool selection and hallucination
detection — four consumers, one number.
"""

from agent_platform.evals import TIER_ORDER, Claim, create_scoring_cascade
from episodes._shared import EpisodeOutput, main

CLAIMS = [
    ("exact value", Claim(text="42", expected="42")),
    ("verbatim in evidence", Claim(text="refund approved", evidence=["the refund approved today"])),
    ("paraphrase", Claim(text="the parcel arrived", evidence=["your package was delivered"])),
    (
        "negated — the trap",
        Claim(text="the refund was approved", evidence=["the refund was denied"]),
    ),
    ("unknowable", Claim(text="the customer was satisfied", evidence=["order 12 shipped"])),
]


async def run() -> EpisodeOutput:
    """Resolve claims through the cascade and report the tier mix."""
    output = EpisodeOutput(
        episode="E10",
        title="NOLLM First",
        principle="Escalating to a model is a failure to resolve cheaply, not a first move.",
    )
    cascade = create_scoring_cascade()

    output.say("[bold]The cascade[/bold]: " + " → ".join(TIER_ORDER))
    output.say("")
    for label, claim in CLAIMS:
        verdict = await cascade.resolve(claim)
        output.say(f"  {label:<22} {verdict.label:<13} via {verdict.tier}")

    output.say("")
    output.say("The negated claim is the one that matters: 'approved' and 'denied'")
    output.say("are near-identical strings and opposite facts. Similarity alone")
    output.say("calls them a match, which is why entailment sits below it.")
    output.say("")
    output.say(
        f"[bold]nollm_scorer_ratio[/bold]: {cascade.nollm_scorer_ratio:.0%} settled without a model"
    )

    return output.model_copy(
        update={
            "artifact": {
                "tiers": len(TIER_ORDER),
                "claims resolved": len(CLAIMS),
                "tier counts": dict(cascade.tier_counts),
                "nollm_scorer_ratio": cascade.nollm_scorer_ratio,
            }
        }
    )


if __name__ == "__main__":
    main(run)
