"""E07 — Orchestrator-Subagent & Agentic Retrieval.

Two shapes, taught at shape level. Orchestration decomposes and delegates
to subagents that cannot see the parent's context — isolation is the
feature and the cost. Agentic retrieval makes looking something up a
decision the model takes mid-reasoning rather than a fixed pipeline stage.

Exit artifact: the depth map — every pattern pointed at the episode that
builds it out in full, which is the map the rest of the series hangs off.
"""

from agentic_commons.context import DenseRetriever, HybridRetriever, LexicalRetriever
from agentic_commons.evals import create_scoring_cascade
from design_patterns import AgenticRetrievalPattern, OrchestratorSubagentPattern, depth_map
from episodes._shared import EpisodeOutput, OfflineLLM, main
from episodes.dummy_data import knowledge_chunks


async def run() -> EpisodeOutput:
    """Run an orchestration and an agent-driven search."""
    output = EpisodeOutput(
        episode="E07",
        title="Orchestrator-Subagent & Agentic Retrieval",
        principle="A subagent sees only its sub-task. Retrieval becomes a decision, not a stage.",
    )

    seen: list[str] = []

    async def decompose(task: str) -> list[str]:
        return ["check stock", "check the price", "check the delivery estimate"]

    async def subagent(subtask: str) -> str:
        seen.append(subtask)
        return f"{subtask} → fine"

    async def synthesize(task: str, results: list[str]) -> str:
        return "; ".join(results)

    orchestrated = await OrchestratorSubagentPattern(decompose, subagent, synthesize).run(
        "assess this order end to end"
    )
    output.say("[bold]Orchestrator-subagent[/bold]")
    for step in orchestrated.steps_of("subagent"):
        output.say(f"  subagent  {step.detail:<26} {step.output}")
    output.say(f"  Each subagent received only its own sub-task: {seen[0]!r}")
    output.say("  It never saw the parent's goal, and could not be distracted by it.")

    corpus = knowledge_chunks()
    retriever = HybridRetriever(
        lexical=LexicalRetriever(corpus),
        dense=DenseRetriever(create_scoring_cascade(), corpus),
    )
    searcher = OfflineLLM(
        ["SEARCH: refund window", "Thirty days from delivery, and damaged items are replaced."]
    )
    retrieved = await AgenticRetrievalPattern(searcher, retriever, max_searches=3).run(
        "how long do I have to return a damaged item?"
    )
    output.say("")
    output.say("[bold]Agentic retrieval[/bold]")
    for step in retrieved.steps:
        output.say(f"  {step.kind:<8} {step.detail}")
    output.say("  The model chose when to search and what to search for.")

    mapping = depth_map()
    return output.model_copy(
        update={
            "artifact": {
                "subagents": orchestrated.metadata["subagents"],
                "searches issued": len(retrieved.steps_of("search")),
                "depth map": {name: ep for name, ep in sorted(mapping.items())},
            }
        }
    )


if __name__ == "__main__":
    main(run)
