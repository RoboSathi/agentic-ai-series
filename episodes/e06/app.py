"""E06 — Planning & Decomposition, Reflection, Evaluator-Optimizer.

Three patterns that look similar and differ in one decisive way: what
checks the work.

Planning commits to a decomposition before acting. Reflection critiques
its own output — and the live failure here is the point: an agent asked
to reflect on a wrong answer produces a *more confident* wrong answer,
because nothing in the loop is external to the model. Evaluator-optimizer
fixes that by scoring against something the generator does not control.
"""

from agentic_commons.evals import create_scoring_cascade
from design_patterns import EvaluatorOptimizerPattern, PlanningPattern, ReflectionPattern
from episodes._shared import EpisodeOutput, OfflineLLM, main


async def run() -> EpisodeOutput:
    """Plan a task, then show reflection failing where an evaluator succeeds."""
    output = EpisodeOutput(
        episode="E06",
        title="Planning, Reflection, Evaluator-Optimizer",
        principle="A generator grading its own work converges on confident, not correct.",
    )

    planner = OfflineLLM(["confirm the order\ncheck the refund window\nissue the refund"])
    executed: list[str] = []

    async def execute(step: str, prior: list[str]) -> str:
        executed.append(step)
        return f"{step}: done"

    plan = await PlanningPattern(planner, execute).run("refund order 55")
    output.say("[bold]Planning[/bold] — commit to a decomposition, then execute it")
    for step in plan.steps_of("execute"):
        output.say(f"  {step.detail}")
    output.say("  A plan is not one autonomy decision. It is a tree of them.")

    critic = OfflineLLM(
        [
            "The refund window is 60 days.",
            "The answer is vague — state it more definitively.",
            "The refund window is definitely 60 days.",
            "NO ISSUES",
        ]
    )
    reflected = await ReflectionPattern(critic, max_rounds=2).run("what is the refund window?")
    output.say("")
    output.say("[bold]Reflection[/bold] — critique your own answer")
    output.say(f"  draft    : {reflected.steps_of('draft')[0].output!r}")
    output.say(f"  revised  : {reflected.output!r}")
    output.say("  More confident. Still wrong — the real window is 30 days.")
    output.say(f"  externally grounded: {reflected.metadata['externally_grounded']}")

    truth = ["Refunds are available for 30 days after delivery"]
    generator = OfflineLLM(
        ["The refund window is 60 days.", "Refunds are available for 30 days after delivery"]
    )
    optimized = await EvaluatorOptimizerPattern(
        generator, create_scoring_cascade(), target_score=0.8, max_rounds=2
    ).run("what is the refund window?", criteria=truth)
    output.say("")
    output.say("[bold]Evaluator-optimizer[/bold] — score against something independent")
    for step in optimized.steps:
        output.say(f"  {step.detail}")
    output.say(f"  → {optimized.output!r}")
    output.say("  The scorer is the shared NOLLM cascade, not the generator.")

    return output.model_copy(
        update={
            "artifact": {
                "plan steps": len(executed),
                "reflection rounds": reflected.metadata["rounds"],
                "reflection grounded": reflected.metadata["externally_grounded"],
                "optimizer best score": optimized.metadata["best_score"],
                "optimizer met target": optimized.succeeded,
            }
        }
    )


if __name__ == "__main__":
    main(run)
