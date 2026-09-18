"""E01 — Agent = Model + Harness.

The claim: what people call "an agent" is a model plus everything around
it, and almost every property you care about in production — whether it
can be audited, bounded, resumed, or stopped — is a property of the
harness, not the model.

This episode makes that concrete by counting. The same question is asked
twice: once with a bare model call, once through the harness. The model
is identical. What differs is everything else.
"""

from agentic_commons.config import get_settings
from agentic_commons.llm.types import CompletionRequest, Message
from episodes._shared import EpisodeOutput, episode_llm, main

# The subsystems a request passes through in the assembled harness. Each
# is a thing the bare call simply does not have.
HARNESS_SUBSYSTEMS = [
    "config",
    "logging",
    "llm",
    "persistence",
    "audit",
    "observability",
    "context",
    "memory",
    "guardrails",
    "hallucination_protection",
    "gatekeeper",
    "security",
    "resilience",
    "evals",
    "tools",
]


async def run() -> EpisodeOutput:
    """Contrast a bare model call with the same call through a harness."""
    output = EpisodeOutput(
        episode="E01",
        title="Agent = Model + Harness",
        principle="The model produces text. The harness is what makes it a system.",
    )
    llm = episode_llm(["An agent is a model plus a harness."])

    result = await llm.complete(
        CompletionRequest(messages=[Message(role="user", content="What is an agent?")])
    )
    output.say(f"[bold]Bare model call[/bold] → {result.content!r}")
    output.say("  no identity, no authority check, no audit trail, no bound on cost")
    output.say("  nothing recorded; ask again tomorrow and nobody can say what happened")

    settings = get_settings()
    output.say("")
    output.say(
        f"[bold]The same call through the harness[/bold] passes {len(HARNESS_SUBSYSTEMS)} subsystems"
    )
    output.say(f"  model            : {settings.llm.providers[settings.llm.provider].model}")
    output.say(
        f"  bounded by       : max_tokens={settings.llm.max_tokens}, timeout={settings.resilience.timeout.default_seconds}s"
    )
    output.say(f"  every decision   : recorded against policy {settings.audit.policy_version}")
    output.say("")
    output.say("Swapping the model changes the answers. Swapping the harness")
    output.say("changes what the system *is*.")

    return output.model_copy(
        update={
            "artifact": {
                "subsystems": len(HARNESS_SUBSYSTEMS),
                "harness": ", ".join(HARNESS_SUBSYSTEMS),
                "bare call guarantees": "none",
                "harness guarantees": "identity, authority, audit, bounds, recovery",
            }
        }
    )


if __name__ == "__main__":
    main(run)
