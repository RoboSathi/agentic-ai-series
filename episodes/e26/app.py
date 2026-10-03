"""E26 — Grounding Failure vs. Reasoning Failure.

One symptom, two causes, opposite fixes.

If the evidence was never retrieved, no amount of prompt work helps — the
fix is retrieval. If the evidence was right there and the model concluded
otherwise, more retrieval makes it worse.

Routing every hallucination to "improve retrieval" is why some of them
never get fixed. The differential diagnosis is the same surface symptom
against two fixtures: evidence present, evidence absent.
"""

from agent_platform.context import Chunk
from agent_platform.hallucination_protection import GroundingReasoningDiagnoser
from episodes._shared import EpisodeOutput, main

CLAIM = "order 55 was refunded $80.00"
PRESENT = [Chunk(content="Order 55 was delivered and refunded $80.00", source="orders")]
ABSENT: list[Chunk] = []
IRRELEVANT = [Chunk(content="The weather forecast for Tuesday is rain", source="noise")]


async def run() -> EpisodeOutput:
    """Same claim, three evidence states, three diagnoses."""
    output = EpisodeOutput(
        episode="E26",
        title="Grounding Failure vs. Reasoning Failure",
        principle="Same symptom, opposite fixes. Naming which one is the whole job.",
    )
    diagnoser = GroundingReasoningDiagnoser()

    cases = [
        ("evidence present", PRESENT, "the retrieval worked; the model still got it wrong"),
        ("evidence absent", ABSENT, "nothing was retrieved; the model had nothing to work from"),
        ("evidence irrelevant", IRRELEVANT, "retrieval ran and returned the wrong thing"),
    ]
    diagnoses = {}
    output.say(f"[bold]The claim[/bold]: {CLAIM!r}")
    output.say("")
    for label, evidence, reading in cases:
        diagnosis = diagnoser.diagnose(CLAIM, evidence)
        diagnoses[label] = diagnosis
        output.say(f"  {label:<20} → [bold]{diagnosis}[/bold] failure")
        output.say(f"    {reading}")

    output.say("")
    output.say("[bold]The fixes are opposite[/bold]")
    output.say("  grounding failure  → fix retrieval: chunking, hybrid search, reranking")
    output.say("  reasoning failure  → fix the model, the prompt, or the task decomposition")
    output.say("")
    output.say("Sending a reasoning failure to the retrieval team wastes a sprint")
    output.say("and comes back unfixed.")

    overlap = diagnoser.overlap(CLAIM, PRESENT[0].content)
    return output.model_copy(
        update={
            "artifact": {
                "claim": CLAIM,
                "diagnoses": diagnoses,
                "overlap with present evidence": overlap,
                "threshold": "grounding_overlap_threshold from config",
            }
        }
    )


if __name__ == "__main__":
    main(run)
