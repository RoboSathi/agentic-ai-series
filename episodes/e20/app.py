"""E20 — Designing Agent Loops: stability, deadlock, livelock.

A loop with no natural stopping point needs one imposed. The watchdog
stack is six layers — step count, wall clock, token budget, cost budget,
duplicate action, semantic stagnation — evaluated in a fixed order so the
*first* layer to trip is the reported reason.

Order matters: a run that blew its budget while also repeating itself
should report the budget, because that is the thing an operator can act
on.

The duplicate-action layer is live in the running product today, and it
was added because a weak model asked for the same order's status twice
and then queued the same refund twice.
"""

from agent_platform.control_loop import LoopVerdict, Trajectory
from agent_platform.tools import ToolCall
from customer_support_agent.control_loop import DuplicateActionDetector, action_fingerprint
from episodes._shared import EpisodeOutput, main

LAYERS = [
    ("step_count", "a hard cap on iterations"),
    ("wall_clock", "a deadline, because a slow loop is still a loop"),
    ("token_budget", "context has a price and a ceiling"),
    ("cost_budget", "the same loop in money rather than tokens"),
    ("duplicate_action", "the same call twice is never useful"),
    ("semantic_stagnation", "different words, no progress"),
]


async def run() -> EpisodeOutput:
    """Show the layer order, then the duplicate detector doing real work."""
    output = EpisodeOutput(
        episode="E20",
        title="Designing Agent Loops",
        principle="The first layer to trip is the reported reason — that ordering is the contract.",
    )

    output.say("[bold]Six watchdog layers, evaluated in order[/bold]")
    for index, (layer, why) in enumerate(LAYERS, start=1):
        output.say(f"  {index}. {layer:<22} {why}")

    detector = DuplicateActionDetector()
    first = ToolCall(action="support.get_order_status", params={"order_id": 55})
    again = ToolCall(action="support.get_order_status", params={"order_id": 55})
    reordered = ToolCall(action="support.process_refund", params={"amount": 80.0, "order_id": 55})
    same_reordered = ToolCall(
        action="support.process_refund", params={"order_id": 55, "amount": 80.0}
    )

    output.say("")
    output.say("[bold]Duplicate action — live in the running product[/bold]")
    detector.record(first, "Order 55: delivered, total $80.00")
    output.say(f"  first call recorded, fingerprint {action_fingerprint(first)}")
    output.say(f"  identical call seen before: {detector.seen_before(again) is not None}")
    output.say(
        f"  argument order ignored    : {action_fingerprint(reordered) == action_fingerprint(same_reordered)}"
    )
    output.say("")
    output.say("  The repeat is answered with the earlier result rather than a")
    output.say("  bare refusal — refusing without the value invites a third attempt:")
    output.say(f"    {detector.reject(again)[:78]}…")

    trajectory = Trajectory(run_id="demo", steps=12, tokens_used=48_000, cost_usd=1.40)
    verdict = LoopVerdict(decision="abort", layer="token_budget", reason="48k over a 30k budget")
    output.say("")
    output.say(
        f"[bold]A verdict[/bold]: {verdict.decision} at layer '{verdict.layer}' — {verdict.reason}"
    )

    return output.model_copy(
        update={
            "artifact": {
                "layers": len(LAYERS),
                "evaluation order": " → ".join(layer for layer, _ in LAYERS),
                "live today": "duplicate_action, step_count",
                "trajectory steps": trajectory.steps,
                "abort layer": verdict.layer,
            }
        }
    )


if __name__ == "__main__":
    main(run)
