"""E25 — Hallucination Is a Measurement Problem First.

"The model hallucinated" is not a finding. Before it can be reduced it
has to be counted, and counting requires deciding what a single claim is.

Scoring a paragraph as a unit hides the case that matters: three correct
sentences and one fabricated number average out to "mostly right", and
the fabricated number is the one that gets acted on. So a generation is
decomposed into atomic claims and each is resolved separately.
"""

from agentic_commons.context import Chunk
from agentic_commons.evals import create_scoring_cascade
from agentic_commons.hallucination_protection import ClaimDecomposer, DetectionLadder
from episodes._shared import EpisodeOutput, main

EVIDENCE = [
    Chunk(content="Order 55 was delivered on 3 March and refunded $80.00.", source="orders")
]
GENERATION = (
    "Order 55 was delivered on 3 March. It was refunded $80.00. "
    "We also sent you a complimentary gift card worth $500."
)


async def run() -> EpisodeOutput:
    """Decompose a generation and score each claim against the evidence."""
    output = EpisodeOutput(
        episode="E25",
        title="Hallucination Is a Measurement Problem",
        principle="Score the paragraph and the fabricated number averages away.",
    )
    ladder = DetectionLadder(create_scoring_cascade())

    claims = ClaimDecomposer().decompose(GENERATION)
    output.say("[bold]One generation, decomposed[/bold]")
    for claim in claims:
        output.say(f"  verifiable={claim.is_verifiable}  {claim.text!r}")

    assessment = await ladder.assess(GENERATION, EVIDENCE)
    output.say("")
    output.say("[bold]Each claim resolved separately[/bold]")
    for verdict in assessment.verdicts:
        mark = "supported" if verdict.supported else "UNSUPPORTED"
        output.say(f"  {mark:<12} via {verdict.tier:<22} {verdict.claim[:44]}")

    output.say("")
    output.say(f"  hallucination rate : {assessment.hallucination_rate:.0%}")
    output.say(f"  settled without a model : {assessment.nollm_resolved_ratio:.0%}")
    output.say("")
    output.say("Scored as one blob this generation is 'mostly right'. Scored as")
    output.say("three claims, one of them invents $500.")

    consistency = ladder.consistency(
        ["the refund was $80.00", "the refund was $80.00", "the refund was $120.00"]
    )
    output.say("")
    output.say(f"[bold]Self-consistency[/bold] across samples: {consistency:.2f}")
    output.say("  A fabricated detail varies between samples; a grounded one does not.")

    return output.model_copy(
        update={
            "artifact": {
                "claims extracted": len(claims),
                "unsupported": assessment.unsupported,
                "hallucination rate": assessment.hallucination_rate,
                "nollm resolved": assessment.nollm_resolved_ratio,
                "consistency": consistency,
            }
        }
    )


if __name__ == "__main__":
    main(run)
