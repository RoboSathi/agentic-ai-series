"""How the nodes are connected, and which of them exist at all.

A withheld feature is a stage that is never built, so it never appears in
the compiled graph. That is the difference worth seeing: an absent control
has no step to narrate, while a merely disabled one still runs, still
traces, and still tells you it found nothing.
"""

from typing import Any

from langgraph.graph import END, StateGraph

from customer_support_agent.graph.loop_decision import after_act_step, after_reason_step
from customer_support_agent.graph.nodes._support import NodeSupport
from customer_support_agent.graph.nodes.review_wait import AWAITING_REVIEW
from customer_support_agent.graph.state import AgentState

# Each optional stage and the feature that decides whether it is built.
_STAGE_FEATURES = {
    "basic_sanity": "basic_sanity",
    "prompt_injection": "prompt_injection",
    "content_moderation": "content_moderation",
    "pii_filtering": "pii",
    "classify_intent": "intent_classification",
    "assemble_context": "context",
}


class GraphWiring(NodeSupport):
    def _build(self) -> Any:
        """Assemble the state graph from the stages this agent was given.

        Only enabled stages become nodes, and the edges are wired to the
        next enabled one — so a withheld capability is absent from the
        compiled graph rather than present and declining to act. That is
        the difference an episode is trying to show: the trace has a gap
        where the control used to be, not a step reporting that it did
        nothing.
        """
        graph: StateGraph[AgentState, None, AgentState, AgentState] = StateGraph(AgentState)
        graph.add_node("authenticate", self.authenticate)
        graph.add_node("choose_order", self.choose_order)
        graph.add_node("read_intent", self.read_intent)
        graph.add_node("reason", self.reason)
        graph.add_node("act", self.act)
        graph.add_node("finalize", self.finalize)

        # The screening-and-routing run, in order, each present only if
        # this deployment declared it.
        front_door = [
            stage
            for stage in (
                # Deliberately first: deterministic domain contradictions are
                # settled from the database before anything spends a model call.
                ("basic_sanity", self.basic_sanity),
                # Before both hosted guards, not after them. The
                # injection and moderation detectors call a provider of
                # their own, so redacting after they ran would mean the
                # customer's email had already left the building twice
                # before the "redact before the LLM call" step reached it.
                ("pii_filtering", self.pii_filtering),
                ("prompt_injection", self.prompt_injection),
                ("content_moderation", self.content_moderation),
                ("classify_intent", self.classify_intent),
                ("assemble_context", self.assemble_context),
            )
            if self._features.is_on(_STAGE_FEATURES[stage[0]]) and self._has_collaborator(stage[0])
        ]
        for name, node in front_door:
            graph.add_node(name, node)

        # Authentication is the hard front-door gate.  An unauthenticated
        # request never reaches security screening, intent classification,
        # retrieval, order selection, or the LLM.
        graph.set_entry_point("authenticate")
        graph.add_conditional_edges(
            "authenticate",
            self._after_authenticate,
            {"choose_order": "choose_order", "done": END},
        )
        # Resolve deterministic menu choices before anything model-driven,
        # then hand to the first stage this agent actually has.
        names = [name for name, _ in front_door]
        first = names[0] if names else "reason"
        graph.add_conditional_edges(
            "choose_order", self._after_choose, {"next": first, "done": END}
        )

        # Each stage continues to the next one that exists. Every one of
        # them can halt the turn, so each edge carries a `done` branch.
        for position, name in enumerate(names):
            following = names[position + 1] if position + 1 < len(names) else "reason"
            if name == "classify_intent":
                graph.add_conditional_edges(
                    name,
                    self._after_classify_intent,
                    {"read_intent": "read_intent", "next": following, "done": END},
                )
            else:
                graph.add_conditional_edges(
                    name, self._after_screening, {"next": following, "done": END}
                )

        graph.add_edge("read_intent", END)
        graph.add_conditional_edges(
            "reason", self._after_reason, {"act": "act", "finalize": "finalize"}
        )
        graph.add_conditional_edges(
            "act", self._after_act, {"reason": "reason", "finalize": "finalize"}
        )
        # A turn that queued a refund for a person pauses after replying,
        # checkpointed, until the decision resumes it. Only a durable agent
        # has the node: an interrupt with nowhere to persist the pause
        # would lose the run it was meant to keep.
        if self._checkpointer is not None:
            graph.add_node("await_approval", self.await_approval)
            graph.add_conditional_edges(
                "finalize", self._after_finalize, {"await": "await_approval", "done": END}
            )
            graph.add_edge("await_approval", END)
        else:
            graph.add_edge("finalize", END)
        return graph.compile(checkpointer=self._checkpointer)

    def _has_collaborator(self, stage: str) -> bool:
        """Whether the object a stage needs was actually injected.

        Belt and braces for a caller that constructs the agent directly:
        a declared feature whose collaborator is missing would otherwise
        compile a node that raises on its first turn.
        """
        required = {
            "basic_sanity": self._basic_sanity,
            "assemble_context": self._context,
            # The suite is always present; its filter is not. Without one
            # there is nothing to redact with, and a stage that narrated
            # "no personal data found" would be reporting the absence of a
            # detector as a clean bill of health.
            "pii_filtering": self._security.pii,
        }
        return required.get(stage, self) is not None

    def _after_finalize(self, state: AgentState) -> str:
        """Pause for a reviewer if this turn queued a refund, else finish."""
        return "await" if state.metadata.get(AWAITING_REVIEW) else "done"

    def _after_screening(self, state: AgentState) -> str:
        """A screening stage either halted the turn or passed it along."""
        return "done" if state.metadata.get("halted") else "next"

    def _after_classify_intent(self, state: AgentState) -> str:
        """Route reads deterministically; keep writes and ambiguity on the LLM path."""
        if state.metadata.get("halted"):
            return "done"
        if state.metadata.get("intent_execution_route") == "deterministic_read":
            return "read_intent"
        return "next"

    def _after_authenticate(self, state: AgentState) -> str:
        """Stop until identity is verified; only verified sessions enter intake.

        A turn that newly supplies a valid customer code ends after the
        authentication read-back.  The next customer turn then enters the
        deterministic order/menu and sanity path before security/intent.
        """
        return "done" if state.metadata.get("halted") else "choose_order"

    def _after_choose(self, state: AgentState) -> str:
        """A menu choice is answered from the database and finishes here."""
        return "done" if state.metadata.get("halted") else "next"

    def _after_reason(self, state: AgentState) -> str:
        """Act on requested tools, or finish. The `reason` node reported why."""
        last = self._current_messages(state)[-1] if self._current_messages(state) else None
        return after_reason_step(last is not None and bool(last.tool_calls))[0]

    def _after_act(self, state: AgentState) -> str:
        """Go round again only while permitted. The `act` node reported why.

        Pure: LangGraph runs a routing function off the event loop, where a
        card cannot be drawn, so it decides and says nothing.
        """
        return after_act_step(
            state.tool_rounds,
            self._max_rounds,
            state.escalated,
            bool(state.metadata.get("refusal_is_final")),
        )[0]
