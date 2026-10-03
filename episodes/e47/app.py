"""E47 — Loose Ends: what doesn't fit a single pillar.

Every real system has decisions that belong to no module and would be
lost if nobody wrote them down. This episode is the register of them.

What is deliberately *not* built is as much a decision as what is. A
parked package with a stated reason is an engineering position; the same
package with no note is an oversight nobody has noticed yet.
"""

from agent_platform import control_loop, multi_agent, prompt_compilation, sandbox
from episodes._shared import EpisodeOutput, main

PARKED = [
    (
        "sandbox",
        sandbox,
        "interface-only",
        "Docker isolation needs a deployment target this repo lacks",
    ),
    (
        "control_loop",
        control_loop,
        "interface-only",
        "FSM + watchdog contracts are normative; the duplicate layer ships in the agent today",
    ),
    (
        "prompt_compilation",
        prompt_compilation,
        "interface-only",
        "DSPy compilation lands as procedural memory (E21)",
    ),
    (
        "multi_agent",
        multi_agent,
        "empty",
        "not designed yet — an interface invented ahead of the design is the one everyone codes against",
    ),
]

DECISIONS = [
    (
        "No `economics` package",
        "cost aggregation lives in observability; a separate package would split one number across two",
    ),
    (
        "Tier-2 cannot block",
        "enforced by the return type, not by review — AdvisoryFlag has no field that stops anything",
    ),
    (
        "Gatekeeper has no enabled flag",
        "an off switch on the mediation point is an execution path around it",
    ),
    (
        "Wire names use `__`",
        "providers reject a dot in a function name; the qualified name stays for logs and policy",
    ),
    (
        "Config split by ownership",
        "the platform ships neutral defaults; each agent overlays its own vocabulary",
    ),
    ("Identity is not a tool argument", "anything in a schema is something the model fills in"),
]


async def run() -> EpisodeOutput:
    """Print the register of parked packages and cross-cutting decisions."""
    output = EpisodeOutput(
        episode="E47",
        title="Loose Ends",
        principle="What is deliberately not built is as much a decision as what is.",
    )

    output.say("[bold]Parked, with reasons[/bold]")
    for name, module, status, why in PARKED:
        exported = len(getattr(module, "__all__", []))
        output.say(f"  {name:<20} {status:<16} {exported} exported")
        output.say(f"    {why}")

    output.say("")
    output.say("[bold]Decisions that belong to no single module[/bold]")
    for decision, why in DECISIONS:
        output.say(f"  {decision}")
        output.say(f"    {why}")

    output.say("")
    output.say("Each of these was a real argument at the time. Written down, they")
    output.say("stop being re-litigated every few months by someone who was not there.")

    return output.model_copy(
        update={
            "artifact": {
                "parked packages": len(PARKED),
                "interface-only": sum(
                    1
                    for *_, s, _ in [(n, m, s, w) for n, m, s, w in PARKED]
                    if s == "interface-only"
                ),
                "cross-cutting decisions": len(DECISIONS),
                "register": "this episode — the place they live",
            }
        }
    )


if __name__ == "__main__":
    main(run)
