"""The scoping graph: /research question → research kind → interview ⇄ user → rubric → approve.

One durable thread per scope. Every turn ends at `await_user`, which
interrupts the graph (`await_decision`) with the reply the user should
read; the next message resumes the same thread. Session phase is graph
state, not a hand-rolled machine: where the thread is paused *is* where
the conversation is.

Slash commands that are conversational acts (`/done`, `/waive`,
`/rubric …`, `/approve`) arrive as ordinary resumed input and are routed
here, so the TUI and the eval `ScriptedUser` drive exactly one path.
"""

from typing import Any

from langgraph.graph import END, START, StateGraph

from agent_platform.durability.hitl import await_decision
from research_agent.graph.scoping_nodes import ScopingNodes
from research_agent.graph.state import ScopingState

WAIT = "await_user"
# The first node: which kind of research the question asks for (or a refusal).
KIND = "research_kind"


def build_scoping_graph(nodes: ScopingNodes, checkpointer: Any, with_rubric: bool) -> Any:
    """Compile the scoping graph; the rubric stages exist only when the feature does."""
    builder = StateGraph(ScopingState)
    builder.add_node(KIND, nodes.kind)
    builder.add_node("interview_turn", nodes.interview)
    builder.add_node("coverage_update", nodes.waive)
    builder.add_node("close_scoping", nodes.close)
    builder.add_node("rubric_frozen", nodes.freeze)
    builder.add_node("say", nodes.say)
    builder.add_node(WAIT, _await_user)
    builder.add_edge(START, KIND)
    builder.add_conditional_edges(KIND, nodes.after_kind, ["interview_turn", END])
    builder.add_edge("interview_turn", WAIT)
    # After a waiver the interviewer moves on to the next open item.
    builder.add_edge("coverage_update", "interview_turn")
    builder.add_edge("say", WAIT)
    builder.add_edge("rubric_frozen", END)
    targets = [
        "interview_turn",
        "coverage_update",
        "close_scoping",
        "rubric_frozen",
        "say",
    ]
    if with_rubric:
        builder.add_node("rubric_generate", nodes.generate_rubric)
        builder.add_node("rubric_edit", nodes.edit_rubric)
        builder.add_edge("close_scoping", "rubric_generate")
        builder.add_edge("rubric_generate", WAIT)
        builder.add_edge("rubric_edit", WAIT)
        targets.append("rubric_edit")
    else:
        # No acceptance criteria to review: the user approves the Brief itself.
        builder.add_node("brief_ready", nodes.brief_ready)
        builder.add_edge("close_scoping", "brief_ready")
        builder.add_edge("brief_ready", WAIT)
    builder.add_conditional_edges(WAIT, nodes.route_input, targets)
    return builder.compile(checkpointer=checkpointer)


async def _await_user(state: ScopingState) -> dict[str, Any]:
    """Pause until the user answers; the reply is what they are shown meanwhile."""
    answer = await_decision(f"user:{state.session_id}", state.reply, phase=state.phase)
    return {"user_input": str(answer)}
