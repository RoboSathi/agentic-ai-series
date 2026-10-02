"""The autonomous run graph (see `run_nodes` for the shape).

Waves are LangGraph supersteps: `wave_start` fans out one `Send` per
ready node, every `research_node` of the wave runs concurrently (bounded
by the scheduler's semaphore), and `wave_done` runs once after the whole
superstep — so a checkpoint lands at every wave boundary and a crash
resumes at the wave it interrupted. With the Independent Audit withheld,
`independent_audit` is simply not a node: synthesis goes straight to
finalization.
"""

from typing import Any

from langgraph.graph import END, START, StateGraph

from research_agent.graph.run_nodes import RunNodes
from research_agent.graph.state import RunState


def build_run_graph(nodes: RunNodes, checkpointer: Any, with_audit: bool) -> Any:
    builder = StateGraph(RunState)
    for name, node in (
        ("init_run", nodes.init_run),
        ("plan_dag", nodes.plan),
        ("wave_start", nodes.wave_start),
        ("research_node", nodes.research_node),
        ("wave_done", nodes.wave_done),
        ("reflect_gap_analysis", nodes.reflect),
        ("synthesize", nodes.synthesize),
        ("gap_fill", nodes.gap_fill),
        ("finalize", nodes.finalize),
        ("archive", nodes.archive),
    ):
        builder.add_node(name, node)
    builder.add_edge(START, "init_run")
    builder.add_edge("init_run", "plan_dag")
    builder.add_conditional_edges("plan_dag", nodes.after_plan, ["wave_start", "finalize"])
    builder.add_conditional_edges(
        "wave_start", nodes.fan_out, ["research_node", "reflect_gap_analysis", "finalize"]
    )
    builder.add_edge("research_node", "wave_done")
    builder.add_edge("wave_done", "wave_start")
    builder.add_conditional_edges(
        "reflect_gap_analysis", nodes.after_reflect, ["wave_start", "synthesize"]
    )
    _audit_edges(builder, nodes, with_audit)
    builder.add_conditional_edges("gap_fill", nodes.after_gap_fill, ["wave_start", "finalize"])
    builder.add_edge("finalize", "archive")
    builder.add_edge("archive", END)
    return builder.compile(checkpointer=checkpointer)


def _audit_edges(builder: StateGraph, nodes: RunNodes, with_audit: bool) -> None:  # type: ignore[type-arg]
    if not with_audit:
        builder.add_edge("synthesize", "finalize")
        return
    builder.add_node("independent_audit", nodes.independent_audit)
    builder.add_edge("synthesize", "independent_audit")
    builder.add_conditional_edges("independent_audit", nodes.after_audit, ["gap_fill", "finalize"])
