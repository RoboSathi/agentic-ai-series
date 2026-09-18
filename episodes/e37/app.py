"""E37 — SRE for Agents: SLOs, error budgets, incident response.

An agent without SLOs is an agent nobody can say is working. The metric
list is configuration, not code: adding an SLO is a config change, and
each target is what the error budget is measured against.

Two targets are deliberately zero — unauthorized actions and policy
violations. A budget of zero means any occurrence is a breach, which is
the correct shape for a control that either holds or does not.
"""

from agentic_commons.config import get_settings
from agentic_commons.observability import SLORegistry
from episodes._shared import EpisodeOutput, main

OBSERVATIONS = {
    "task_success_rate": [1.0] * 18 + [0.0, 0.0],
    "hallucination_rate": [0.0] * 19 + [1.0],
    "escalation_rate": [0.0] * 15 + [1.0] * 5,
    "unauthorized_action_rate": [0.0] * 20,
    "tool_failure_rate": [0.0] * 18 + [1.0, 1.0],
}


async def run() -> EpisodeOutput:
    """Record observations and report error budgets."""
    output = EpisodeOutput(
        episode="E37",
        title="SRE for Agents",
        principle="A budget of zero means any occurrence is a breach — the right shape for a control.",
    )
    registry = SLORegistry()
    targets = get_settings().observability.slo_targets

    for metric, values in OBSERVATIONS.items():
        for value in values:
            registry.record(metric, value)

    output.say(
        f"[bold]{len(targets)} SLOs configured[/bold], {len(OBSERVATIONS)} observed this run"
    )
    output.say("")
    output.say(f"  {'metric':<26} {'target':>8} {'observed':>9} {'budget left':>12}")
    for metric in OBSERVATIONS:
        status = registry.status(metric)
        output.say(
            f"  {metric:<26} {status.target:>8.2f} {status.observed:>9.2f} "
            f"{status.error_budget_remaining:>11.0%}  {'ok' if status.within_target else 'BREACH'}"
        )

    breaching = registry.breaching()
    output.say("")
    output.say(f"[bold]Breaching[/bold]: {[status.metric for status in breaching] or 'none'}")

    zero_budget = [metric for metric, target in targets.items() if target == 0.0]
    output.say("")
    output.say(f"[bold]Zero-tolerance metrics[/bold]: {', '.join(zero_budget)}")
    output.say("  These do not get a budget to spend. One unauthorized action is")
    output.say("  not 'within tolerance' — it is the thing the control exists for.")

    return output.model_copy(
        update={
            "artifact": {
                "slos configured": len(targets),
                "breaching": ", ".join(s.metric for s in breaching) or "none",
                "zero tolerance": ", ".join(zero_budget),
                "task success": registry.observed("task_success_rate"),
            }
        }
    )


if __name__ == "__main__":
    main(run)
