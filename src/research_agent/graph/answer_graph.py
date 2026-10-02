"""The direct answer's graph, compiled from its nodes.

    START -> reason ⇄ act -> ground -> finalize -> END

No checkpointer: a single question has nothing to resume.
"""

from typing import Any

from langgraph.graph import END, START, StateGraph

from research_agent.graph.answer_nodes import AnswerNodes
from research_agent.graph.answer_state import AnswerState


def build_answer_graph(nodes: AnswerNodes) -> Any:
    builder = StateGraph(AnswerState)
    builder.add_node("reason", nodes.reason)
    builder.add_node("act", nodes.act)
    builder.add_node("ground", nodes.ground)
    builder.add_node("finalize", nodes.finalize)
    builder.add_edge(START, "reason")
    builder.add_conditional_edges("reason", nodes.after_reason, ["act", "ground", "finalize"])
    builder.add_conditional_edges("act", nodes.after_act, ["reason", "finalize"])
    builder.add_edge("ground", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile()


__all__ = ["build_answer_graph"]
