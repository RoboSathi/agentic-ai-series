"""E09 — The Eval Pyramid: from unit invariant to production shadow.

Four levels, four cadences. Component evals run on every change;
production shadow runs continuously against real traffic. The shape
matters: a suite that is all end-to-end tells you something broke without
telling you what, and one that is all unit tests tells you nothing broke
while the system fails.

Exit artifact: the eval manifest — a checked-in test plan, not a
spreadsheet. A component absent from it has no regression signal, and the
manifest is what makes that absence visible.
"""

from agentic_commons.evals import ComponentEvalRunner, EvalCase, load_manifest
from episodes._shared import EpisodeOutput, main

LEVELS = {
    "component": "one component alone — every change",
    "integration": "components wired together — every pull request",
    "golden_task": "whole tasks, curated set — nightly",
    "production_shadow": "real traffic, scored but not served — continuously",
}


async def run() -> EpisodeOutput:
    """Print the pyramid and run the component level over a small suite."""
    output = EpisodeOutput(
        episode="E09",
        title="The Eval Pyramid",
        principle="The eval plan is code, checked in — not a spreadsheet nobody reads.",
    )
    manifest = load_manifest()

    output.say("[bold]Four levels, four cadences[/bold]")
    for level, description in LEVELS.items():
        declared = len(manifest.for_level(level))  # type: ignore[arg-type]
        output.say(f"  {level:<18} {declared:>2} declared   {description}")

    output.say("")
    output.say("[bold]Running the component level[/bold]")
    cases = [
        EvalCase(case_id="c1", component="evals.scoring", input="q", expected="delivered"),
        EvalCase(case_id="c2", component="evals.scoring", input="q", expected="refunded"),
    ]

    async def system(case: EvalCase) -> str:
        return "delivered"

    report = await ComponentEvalRunner().run(cases, system)
    for outcome in report.outcomes:
        mark = "pass" if outcome.passed else "FAIL"
        output.say(f"  {outcome.case_id}  {mark:<4} via {outcome.tier}")
    output.say(f"  pass rate {report.pass_rate:.0%}, threshold {report.regression_threshold:.0%}")

    gaps = manifest.missing_coverage(["evals.scoring", "gatekeeper.authorization", "invented"])
    output.say("")
    output.say(f"Components with no declared eval: {gaps or 'none'}")
    output.say("The gap is meant to be visible here, not discovered in production.")

    return output.model_copy(
        update={
            "artifact": {
                "manifest entries": len(manifest.entries),
                "components covered": len(manifest.components()),
                "levels": ", ".join(LEVELS),
                "nollm ratio this run": report.nollm_scorer_ratio,
            }
        }
    )


if __name__ == "__main__":
    main(run)
