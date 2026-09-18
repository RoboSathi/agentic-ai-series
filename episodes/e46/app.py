"""E46 — The Economics of Autonomy: ROI, TCO & the cancellation case.

Most agent business cases count the model and stop. The costs that
actually decide the outcome are the ones downstream of autonomy:
escalation review, evals, incident response, and the engineering to keep
a harness honest.

This episode does the arithmetic honestly, including the case where the
right answer is to cancel — which is a real outcome and should be
reachable by the same model that justifies building.
"""

from agentic_commons.observability import LittlesLawQueueModel
from episodes._shared import EpisodeOutput, main

VOLUME_PER_MONTH = 20_000
MINUTES_PER_MANUAL_TICKET = 6.0
REVIEWER_COST_PER_HOUR = 28.0
MODEL_COST_PER_REQUEST = 0.004
ESCALATION_RATES = [0.05, 0.20, 0.60]
# Reviewing an escalation costs *more* than handling the ticket outright.
# The reviewer has to reconstruct what the agent did before they can
# agree with it, and that reading is not free. Business cases that assume
# review is cheaper than the work are the ones that disappoint.
REVIEW_MINUTES = 9.0
HOURS_PER_MONTH = 160.0
# Evals, incident response and keeping the harness honest. Real, ongoing,
# and absent from most agent business cases.
PLATFORM_COST_PER_MONTH = 12_000.0


async def run() -> EpisodeOutput:
    """Run the business case at three escalation rates."""
    output = EpisodeOutput(
        episode="E46",
        title="The Economics of Autonomy",
        principle="Count the escalation queue, or the business case is fiction.",
    )
    model = LittlesLawQueueModel()

    manual_hours = VOLUME_PER_MONTH * MINUTES_PER_MANUAL_TICKET / 60
    manual_cost = manual_hours * REVIEWER_COST_PER_HOUR
    output.say(f"[bold]Today, fully manual[/bold]: {VOLUME_PER_MONTH:,} tickets/month")
    output.say(f"  {manual_hours:,.0f} reviewer-hours  ${manual_cost:,.0f}/month")
    output.say("")
    output.say(
        f"[bold]Automated[/bold] — plus ${PLATFORM_COST_PER_MONTH:,.0f}/month of evals, ops and"
    )
    output.say(f"engineering, and {REVIEW_MINUTES:.0f} minutes per escalation (longer than")
    output.say(
        f"the {MINUTES_PER_MANUAL_TICKET:.0f} it takes to just do it — the reviewer must first"
    )
    output.say("reconstruct what the agent did).")

    output.say("")
    output.say(
        f"  {'escalation':>11} {'model $':>10} {'review hrs':>11} {'review $':>10} {'total':>10} {'saving':>10}"
    )
    results = {}
    for rate in ESCALATION_RATES:
        escalated = VOLUME_PER_MONTH * rate
        arrivals_per_hour = escalated / HOURS_PER_MONTH
        review_hours = model.required_capacity(arrivals_per_hour, REVIEW_MINUTES) * HOURS_PER_MONTH
        review_cost = review_hours * REVIEWER_COST_PER_HOUR
        model_cost = VOLUME_PER_MONTH * MODEL_COST_PER_REQUEST
        total = model_cost + review_cost + PLATFORM_COST_PER_MONTH
        saving = manual_cost - total
        results[f"{rate:.0%}"] = round(saving)
        output.say(
            f"  {rate:>10.0%} ${model_cost:>9,.0f} {review_hours:>11,.0f} "
            f"${review_cost:>9,.0f} ${total:>9,.0f} ${saving:>9,.0f}"
        )

    output.say("")
    output.say("The model is the smallest number in every row. The escalation rate")
    output.say("decides whether this is worth doing — and it is a property of the")
    output.say("harness's confidence, not of the model's price.")

    worst = min(results.values())
    output.say("")
    if worst < 0:
        output.say("[bold]The cancellation case[/bold]: at the highest escalation rate this")
        output.say("loses money. That is a real outcome, and a business case that")
        output.say("cannot produce it is not a business case.")

    return output.model_copy(
        update={
            "artifact": {
                "monthly volume": f"{VOLUME_PER_MONTH:,}",
                "manual baseline": f"${manual_cost:,.0f}/month",
                "saving by escalation rate": results,
                "dominant cost": "human review, not inference",
                "cancellation reachable": worst < 0,
            }
        }
    )


if __name__ == "__main__":
    main(run)
