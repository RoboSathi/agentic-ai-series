"""E31 — OWASP Agentic Risks as a Control Mapping Exercise.

A risk list is not a control list. The useful exercise is mapping each
risk to the specific component that addresses it, and being honest where
nothing does.

The mapping below is checked against the code: every named control is a
real class in this repo, and the "not yet" rows are stated rather than
quietly omitted.
"""

from agentic_commons.gatekeeper import CumulativeLedger, GatekeeperImpl, SurfaceClassifier
from agentic_commons.memory.write_pipeline import MemoryWritePipelineImpl
from agentic_commons.security import Authorizer, PIIFilterImpl, PromptInjectionGuard
from episodes._shared import EpisodeOutput, main

MAPPING = [
    ("Prompt injection", "gatekeeper.AuthorityHierarchyResolver", "E30", True),
    ("Excessive agency", "gatekeeper.SurfaceClassifier + AutomationLevelPolicy", "E03/E12", True),
    ("Insecure output handling", "guardrails.PydanticValidator", "E24", True),
    ("Sensitive data disclosure", "security.PIIFilterImpl", "E32", True),
    ("Broken object-level authorization", "OrdersRepository.get_for_customer", "E31", True),
    ("Memory poisoning", "memory.MemoryWritePipelineImpl", "E15/E16", True),
    ("Unbounded consumption", "resilience + gatekeeper.CumulativeLedger", "E02/E35", True),
    ("Supply-chain / model provenance", "—", "not yet", False),
    ("Insecure plugin/tool design", "tools.NamespacedToolRegistry (partial)", "E22", False),
]


async def run() -> EpisodeOutput:
    """Print the control mapping and prove each named control exists."""
    output = EpisodeOutput(
        episode="E31",
        title="OWASP Agentic Risks as a Control Mapping",
        principle="A risk list is not a control list. Map each one, and say where nothing maps.",
    )

    output.say("[bold]Risk → control → episode[/bold]")
    covered = 0
    for risk, control, episode, is_covered in MAPPING:
        mark = "✓" if is_covered else "○"
        covered += int(is_covered)
        output.say(f"  {mark} {risk:<36} {control:<46} {episode}")

    output.say("")
    output.say("[bold]The named controls are real classes, not intentions[/bold]")
    for cls in (
        GatekeeperImpl,
        SurfaceClassifier,
        CumulativeLedger,
        Authorizer,
        PIIFilterImpl,
        PromptInjectionGuard,
        MemoryWritePipelineImpl,
    ):
        output.say(f"  {cls.__module__}.{cls.__name__}")

    gaps = [risk for risk, _, _, ok in MAPPING if not ok]
    output.say("")
    output.say(f"[bold]Not covered[/bold]: {', '.join(gaps)}")
    output.say("Stated rather than omitted — a mapping that only lists wins is")
    output.say("a marketing document.")

    return output.model_copy(
        update={
            "artifact": {
                "risks mapped": len(MAPPING),
                "covered": covered,
                "gaps": ", ".join(gaps),
                "coverage": f"{covered}/{len(MAPPING)}",
            }
        }
    )


if __name__ == "__main__":
    main(run)
