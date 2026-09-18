"""E04 — Workflow, Agent, or Hybrid?

The choice is not about sophistication. It is about who decides the
control flow: a person at design time (WF) or a model at run time (AG).

The Three Gates decide: how many paths lead to the outcome, how far the
damage reaches if it is wrong, and whether the result can be checked
afterwards. An open-ended path with a wide blast radius and no
certifiable result is the profile that should not be automated — and
naming *which* gate failed turns "this feels risky" into an argument.

Exit artifact: the pattern-selection decision record.
"""

from agentic_commons.gatekeeper import ThreeGatesEvaluator
from design_patterns import ALL_PATTERNS, by_family, depth_map, requiring_human_oversight
from episodes._shared import EpisodeOutput, main

CANDIDATES = [
    ("look up an order", "single", "single_record", True),
    ("refund under the threshold", "bounded", "account", True),
    ("decide a disputed refund", "open", "account", False),
    ("rewrite the refund policy", "open", "global", False),
]


async def run() -> EpisodeOutput:
    """Score candidate tasks through the Three Gates and print the record."""
    output = EpisodeOutput(
        episode="E04",
        title="Workflow, Agent, or Hybrid?",
        principle="Who decides the control flow — a person at design time, or a model at run time.",
    )
    gates = ThreeGatesEvaluator()

    output.say("[bold]The Three Gates[/bold] — path cardinality × blast radius × certifiability")
    for task, cardinality, radius, certifiable in CANDIDATES:
        result = gates.evaluate(cardinality, radius, certifiable)  # type: ignore[arg-type]
        mark = "pass" if result.passes else "FAIL"
        output.say(f"  {task:<28} {mark:>4}  — {result.reason}")

    workflow, agentic = by_family("WF"), by_family("AG")
    output.say("")
    output.say(f"[bold]The catalogue[/bold]: {len(workflow)} workflow, {len(agentic)} agentic")
    output.say(f"  WF: {', '.join(card.name for card in workflow)}")
    output.say(f"  AG: {', '.join(card.name for card in agentic)}")

    oversight = requiring_human_oversight()
    output.say("")
    output.say("Patterns the gates say should not run unattended:")
    for card in oversight:
        output.say(f"  {card.name} — {card.avoid_when}")

    return output.model_copy(
        update={
            "artifact": {
                "patterns": len(ALL_PATTERNS),
                "workflow": len(workflow),
                "agentic": len(agentic),
                "needing oversight": ", ".join(card.name for card in oversight),
                "depth map": f"{len(depth_map())} patterns mapped to the episode that builds each out",
            }
        }
    )


if __name__ == "__main__":
    main(run)
