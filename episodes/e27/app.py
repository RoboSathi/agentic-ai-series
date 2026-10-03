"""E27 — Trust Calibration: automation bias, human review, handoff.

Two opposite failures, both expensive.

Automation bias is accepting a wrong answer because the machine produced
it. Algorithm aversion is rejecting a right one for the same reason. A
review step that does either consistently is not a control — it is
theatre with a salary attached.

Only measurement distinguishes them, and the measurement is cheap: for
each reviewed item, was the agent right, and did the human accept it?
"""

from agent_platform.hallucination_protection import TrustCalibrationTracker
from episodes._shared import EpisodeOutput, main

# (agent_correct, human_accepted) for twenty reviewed items.
REVIEWS = [
    (True, True),
    (True, True),
    (True, True),
    (True, True),
    (True, True),
    (True, True),
    (True, False),
    (True, False),
    (True, False),
    (False, True),
    (False, True),
    (False, True),
    (False, False),
    (False, False),
    (True, True),
    (True, True),
    (False, True),
    (True, True),
    (True, False),
    (False, False),
]


async def run() -> EpisodeOutput:
    """Measure reliance over a sample of reviewed decisions."""
    output = EpisodeOutput(
        episode="E27",
        title="Trust Calibration",
        principle="A reviewer who always agrees, and one who never does, are equally useless.",
    )
    tracker = TrustCalibrationTracker()
    for correct, accepted in REVIEWS:
        tracker.record(agent_correct=correct, human_accepted=accepted)

    report = tracker.report()
    output.say(f"[bold]{report.samples} reviewed decisions[/bold]")
    output.say(
        f"  automation bias     : {report.automation_bias_rate:.0%}  (accepted a wrong answer)"
    )
    output.say(
        f"  algorithm aversion  : {report.algorithm_aversion_rate:.0%}  (rejected a right one)"
    )
    output.say(f"  appropriate reliance: {report.appropriate_reliance_rate:.0%}")

    output.say("")
    output.say("[bold]What each failure costs[/bold]")
    output.say("  bias     → the control exists on paper and catches nothing")
    output.say("  aversion → the agent's correct work is discarded; the automation")
    output.say("             never pays back, and looks like it never could")

    output.say("")
    output.say("Neither is visible from the agent's accuracy alone. Both require")
    output.say("recording what the *human* did with each answer.")

    return output.model_copy(
        update={
            "artifact": {
                "samples": report.samples,
                "automation bias": report.automation_bias_rate,
                "algorithm aversion": report.algorithm_aversion_rate,
                "appropriate reliance": report.appropriate_reliance_rate,
                "feeds": "observability.SLORegistry",
            }
        }
    )


if __name__ == "__main__":
    main(run)
