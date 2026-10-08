"""The ops agent's graph.

    START -> reason ⇄ act -> ground -> END

No checkpointer: one question has nothing to resume.
"""

from typing import Any

from langgraph.graph import END, START, StateGraph

from ops_agent.graph.nodes import ACT, GROUND, OpsNodes
from ops_agent.graph.state import OpsState

REASON = "reason"


def build_graph(nodes: OpsNodes) -> Any:
    builder = StateGraph(OpsState)
    builder.add_node(REASON, nodes.reason)
    builder.add_node(ACT, nodes.act)
    builder.add_node(GROUND, nodes.ground)
    builder.add_edge(START, REASON)
    builder.add_conditional_edges(REASON, nodes.after_reason, [ACT, GROUND])
    builder.add_edge(ACT, REASON)
    builder.add_edge(GROUND, END)
    return builder.compile()


__all__ = ["build_graph"]
