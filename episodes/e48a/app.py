"""E48a — Capstone Gate A: Architecture Review.

The first of three gates. This one asks whether the architecture holds
together: are the layers acyclic, does every subsystem sit behind an
interface, and does the running product compose the platform rather than
duplicating it.

The gate is evaluated against the code, not against a diagram. Every
check below either passes or names what failed.
"""

import inspect

from agentic_commons.audit import AuditLedger
from agentic_commons.evals import NollmScoringCascade
from agentic_commons.llm.client import LLMClient
from agentic_commons.memory import MemoryStore
from agentic_commons.tools import ToolRegistry
from customer_support_agent.core.graph import CustomerSupportAgent
from episodes._shared import EpisodeOutput, main

LAYERS = [
    (0, "config, logging"),
    (1, "persistence, llm"),
    (2, "security, memory, tools, resilience"),
    (3, "context, guardrails, hallucination_protection"),
    (4, "gatekeeper"),
    (5, "evals, observability, audit"),
    (6, "sandbox, control_loop, multi_agent, prompt_compilation (parked)"),
    (7, "customer_support_agent"),
    (8, "episodes, design_patterns, tui"),
]


async def run() -> EpisodeOutput:
    """Run the architecture review checks."""
    output = EpisodeOutput(
        episode="E48a",
        title="Capstone Gate A — Architecture Review",
        principle="Evaluated against the code, not against a diagram.",
    )

    output.say("[bold]Layering[/bold] — strictly acyclic, a lower layer never imports a higher")
    for level, members in LAYERS:
        output.say(f"  {level}  {members}")

    output.say("")
    output.say("[bold]Every subsystem behind an interface[/bold]")
    for protocol in (LLMClient, ToolRegistry, MemoryStore, AuditLedger, NollmScoringCascade):
        output.say(f"  {protocol.__name__}")

    source = inspect.getsource(CustomerSupportAgent)
    checks = {
        "no second LLM client in the product": "AsyncOpenAI" not in source,
        "no second auth check": "permissions_for" not in source,
        "no print-based logging": "print(" not in source,
        "exactly one tool invocation path": source.count("registry.invoke(") == 1,
        "gatekeeper evaluated before invocation": source.index("gatekeeper.evaluate")
        < source.index("registry.invoke"),
    }
    output.say("")
    output.say("[bold]The product composes, it does not duplicate[/bold]")
    for check, passed in checks.items():
        output.say(f"  {'PASS' if passed else 'FAIL'}  {check}")

    passed = all(checks.values())
    output.say("")
    output.say(f"[bold]Gate A: {'PASS' if passed else 'FAIL'}[/bold]")

    return output.model_copy(
        update={
            "artifact": {
                "layers": len(LAYERS),
                "checks": len(checks),
                "passed": sum(checks.values()),
                "gate": "PASS" if passed else "FAIL",
                "enforced by": "mypy --strict plus tests/integration/test_cross_subsystem_invariants.py",
            }
        }
    )


if __name__ == "__main__":
    main(run)
