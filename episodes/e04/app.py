"""E04 — Workflow, Agent, or Hybrid?

The choice is not about sophistication. It is about who decides the
control flow: a person at design time (WF) or a model at run time (AG).

That is a decision, and a decision needs a record. A catalogue entry
therefore says both when a shape fits and when it does not — `avoid_when`
is the half that matters at 2am, because the reason a pattern was chosen
is rarely the reason it later hurt.

Exit artifact: the pattern-selection decision record.
"""

from design_patterns import ALL_PATTERNS, by_family, depth_map
from episodes._shared import EpisodeOutput, main


async def run() -> EpisodeOutput:
    """Print the catalogue as a decision record, split by who decides the path."""
    output = EpisodeOutput(
        episode="E04",
        title="Workflow, Agent, or Hybrid?",
        principle="Who decides the control flow — a person at design time, or a model at run time.",
    )

    workflow, agentic = by_family("WF"), by_family("AG")
    output.say(f"[bold]The catalogue[/bold]: {len(workflow)} workflow, {len(agentic)} agentic")
    output.say("")

    output.say("[bold]WF[/bold] — a person fixed the path before anything ran")
    for card in workflow:
        output.say(f"  {card.name:<24} use when {card.use_when}")

    output.say("")
    output.say("[bold]AG[/bold] — a model chooses the path while it runs")
    for card in agentic:
        output.say(f"  {card.name:<24} use when {card.use_when}")

    # The half of the record nobody writes down, and the only half that
    # is any use once the pattern is already in production.
    output.say("")
    output.say("[bold]And when not to[/bold]")
    for card in ALL_PATTERNS:
        output.say(f"  {card.name:<24} avoid when {card.avoid_when}")

    return output.model_copy(
        update={
            "artifact": {
                "patterns": len(ALL_PATTERNS),
                "workflow": len(workflow),
                "agentic": len(agentic),
                "depth map": (
                    f"{len(depth_map())} patterns mapped to the episode that builds each out"
                ),
            }
        }
    )


if __name__ == "__main__":
    main(run)
