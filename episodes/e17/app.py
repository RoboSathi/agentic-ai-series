"""E17 — Context Failure: rot, distraction, confusion, clash, poisoning.

Five distinct failure modes, which matters because they have different
fixes. Treating them all as "the context is too long" leads to trimming a
window whose actual problem was that two chunks contradicted each other.

The regression test is the degradation curve: measured at configured
token checkpoints against a short-window baseline, so a drop is caught
rather than assumed from a published benchmark.
"""

from agent_platform.context import AssembledContext, Chunk, ContextFailureDetector
from episodes._shared import EpisodeOutput, main

MODES = {
    "rot": "quality decays as the window grows, with nothing wrong in it",
    "distraction": "irrelevant material pulls the answer off task",
    "confusion": "too many similar options; the model cannot choose",
    "clash": "the window contains contradictory facts",
    "poisoning": "a false statement in context is treated as established",
}


async def run() -> EpisodeOutput:
    """Detect each mode, then plot the degradation curve."""
    output = EpisodeOutput(
        episode="E17",
        title="Context Failure",
        principle="Five failure modes, five different fixes. 'Too long' is only one of them.",
    )
    detector = ContextFailureDetector()

    output.say("[bold]The five modes[/bold]")
    for mode, description in MODES.items():
        output.say(f"  {mode:<13} {description}")

    clashing = AssembledContext(
        chunks=[
            Chunk(content="status: shipped", source="cache", token_estimate=10),
            Chunk(content="status: cancelled", source="live", token_estimate=10),
        ],
        budget_tokens=100,
        used_tokens=20,
    )
    contradictions = detector.contradictions(clashing)
    output.say("")
    output.say("[bold]Clash detected structurally[/bold]")
    for key, values in contradictions:
        output.say(f"  '{key}' has conflicting values: {values}")
    output.say("  A stale cache disagreeing with a live tool result — the case")
    output.say("  that actually happens, caught without a model.")

    distracting = AssembledContext(
        chunks=[
            Chunk(content="completely unrelated marketing copy"),
            Chunk(content="the refund policy is thirty days"),
        ]
    )
    ratio = detector.distraction_ratio(distracting, {"refund"})
    output.say("")
    output.say(f"[bold]Distraction[/bold]: {ratio:.0%} of chunks share no term with the task")

    curve = detector.degradation_curve({10_000: 0.92, 30_000: 0.91, 60_000: 0.88, 150_000: 0.61})
    output.say("")
    output.say("[bold]The degradation curve[/bold] — measured, not assumed")
    for health in curve:
        mark = "DEGRADED" if health.degraded else "ok"
        output.say(
            f"  {health.checkpoint_tokens:>7,} tokens  score {health.observed_score:.2f}  {mark:<9} {health.detail}"
        )

    return output.model_copy(
        update={
            "artifact": {
                "modes": ", ".join(MODES),
                "checkpoints": detector.checkpoints,
                "clashes found": len(contradictions),
                "distraction ratio": ratio,
                "first degraded checkpoint": next(
                    (h.checkpoint_tokens for h in curve if h.degraded), None
                ),
            }
        }
    )


if __name__ == "__main__":
    main(run)
