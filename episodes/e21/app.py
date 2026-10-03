"""E21 — Prompt Engineering as Software.

A hand-tuned prompt string is an artifact nobody can re-derive: somebody
adjusted it until the demo worked, and the reasoning left with them.

The alternative is a typed signature plus an optimizer, so the wording
becomes a compiled artifact with a version and an eval score. Which makes
it procedural memory — versioned, rollback-able, exactly like the how-to
artifacts in E14 — rather than a new kind of thing.

`prompt_compilation` is interface-only this phase; this episode shows the
contract and the procedural-memory link it lands in.
"""

from agent_platform.memory import MemoryFact, ProceduralStore, Provenance
from agent_platform.persistence import IN_MEMORY, SqliteConnectionPool
from agent_platform.prompt_compilation import CompiledModule, Example, Optimizer, Signature
from episodes._shared import EpisodeOutput, main


async def run() -> EpisodeOutput:
    """Show the contract, then version a prompt as procedural memory."""
    output = EpisodeOutput(
        episode="E21",
        title="Prompt Engineering as Software",
        principle="A compiled prompt is a procedural-memory artifact, not a new kind of thing.",
    )

    output.say("[bold]The contract[/bold] (interface-only this phase)")
    for protocol, purpose in [
        (Signature, "typed input→output contract, independent of wording"),
        (Optimizer, "compiles a signature + metric + trainset into a module"),
        (CompiledModule, "a versioned, runnable prompt"),
        (Example, "one training example"),
    ]:
        output.say(f"  {protocol.__name__:<16} {purpose}")

    output.say("")
    output.say("Callers depend on the signature, so the wording underneath can be")
    output.say("recompiled without any caller changing.")

    pool = SqliteConnectionPool(IN_MEMORY)
    store = ProceduralStore(pool)
    curator = Provenance(source="optimizer", kind="agent_inference")
    for version, body in enumerate(
        [
            "Answer the customer's refund question.",
            "Answer the customer's refund question. Confirm the amount first.",
            "Confirm the amount, then answer. Never guess a threshold.",
        ],
        start=1,
    ):
        store.write(
            MemoryFact(
                key="refund_prompt",
                value=body,
                metadata={"artifact_type": "compiled_prompt", "eval_score": 0.7 + version * 0.05},
                provenance=curator,
            ),
            curator,
        )

    history = store.history("refund_prompt")
    active = store.get("refund_prompt")
    output.say("")
    output.say("[bold]The same prompt, versioned as procedural memory[/bold]")
    for fact in reversed(history):
        marker = (
            "←active" if active and fact.metadata["version"] == active.metadata["version"] else ""
        )
        output.say(f"  v{fact.metadata['version']}  {fact.value[:52]:<54} {marker}")

    store.rollback("refund_prompt", 2)
    rolled = store.get("refund_prompt")
    output.say("")
    output.say(f"  rollback to v2 → {rolled.value[:52]!r}" if rolled else "")
    output.say("  A prompt you cannot roll back is a prompt you cannot ship safely.")

    return output.model_copy(
        update={
            "artifact": {
                "status": "interface-only; implementation is next phase",
                "contracts": "Signature, Optimizer, CompiledModule, Example",
                "versions stored": len(history),
                "active after rollback": rolled.metadata["version"] if rolled else None,
                "lands in": "memory.ProceduralStore (E14) + the deployment registry (E39)",
            }
        }
    )


if __name__ == "__main__":
    main(run)
