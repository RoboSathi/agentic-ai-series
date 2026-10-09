"""The ops agent's graph.

    START -> propose ⇄ execute -> ground -> END

No checkpointer: one question has nothing to resume.
"""

from typing import Any

from langgraph.graph import END, START, StateGraph

from ops_agent.graph.nodes import EXECUTE, GROUND, PROPOSE, OpsNodes
from ops_agent.graph.state import OpsState


def build_graph(nodes: OpsNodes) -> Any:
    builder = StateGraph(OpsState)
    builder.add_node(PROPOSE, nodes.propose)
    builder.add_node(EXECUTE, nodes.execute)
    builder.add_node(GROUND, nodes.ground)
    builder.add_edge(START, PROPOSE)
    builder.add_conditional_edges(PROPOSE, nodes.after_propose, [EXECUTE, GROUND])
    builder.add_edge(EXECUTE, PROPOSE)
    builder.add_edge(GROUND, END)
    return builder.compile()


__all__ = ["build_graph"]
