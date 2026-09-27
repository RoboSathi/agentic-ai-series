"""E03 — Autonomy Is a Control Allocation, Not a Feature Toggle.

"Human in the loop" names no point on any scale. Sheridan & Verplank's
1978 levels run 1 to 10, and the useful question is always *which level*,
for *which action*, at *what amount* — because the same tool is a
different decision at different sizes.

The demonstration: one action, three amounts, three different answers,
all from the same configured policy.
"""

from agentic_commons.gatekeeper import SHERIDAN_LEVELS, AutomationLevelPolicy
from customer_support_agent.admin.refund_limit import current_refund_limit
from episodes._shared import EpisodeOutput, main

AMOUNTS = [10.0, 49.0, 50.0, 51.0, 500.0]


async def run() -> EpisodeOutput:
    """Show automation level varying by action and by amount."""
    output = EpisodeOutput(
        episode="E03",
        title="Autonomy Is a Control Allocation",
        principle="Not whether a human is involved, but which level, for which action.",
    )
    policy = AutomationLevelPolicy()

    output.say("[bold]The same action, at different amounts[/bold]")
    for amount in AMOUNTS:
        assessment = policy.assess("process_refund", amount)
        verdict = "human required" if assessment.requires_human else "autonomous"
        output.say(f"  process_refund ${amount:>7,.2f}  →  {verdict}")

    output.say("")
    output.say("[bold]Different actions sit at different levels[/bold]")
    levels = {}
    for action in (
        "get_order_status",
        "queue_for_approval",
        "process_refund",
    ):
        level = policy.level_for(action)
        levels[action] = level
        output.say(f"  {action:<22} level {level:>2}  {SHERIDAN_LEVELS[level]}")

    output.say("")
    output.say("None of this is a boolean. Moving an action from level 7 to 5")
    output.say("is a decision someone makes, writes down, and can defend.")

    return output.model_copy(
        update={
            "artifact": {
                "scale": "Sheridan & Verplank 1978, levels 1-10",
                "levels by action": levels,
                "escalation amount": f"{current_refund_limit()} for process_refund",
                "decision": "per action, per amount — not a feature toggle",
            }
        }
    )


if __name__ == "__main__":
    main(run)
