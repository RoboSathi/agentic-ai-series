"""How the nodes are connected, and which of them exist at all.

A withheld feature is a stage that is never built, so it never appears in
the compiled graph. That is the difference worth seeing: an absent control
has no step to narrate, while a merely disabled one still runs, still
traces, and still tells you it found nothing.
"""

from typing import Any

from langgraph.graph import END, StateGraph

from customer_support_agent.graph.nodes._support import NodeSupport
from customer_support_agent.graph.state import AgentState
from customer_support_agent.presentation.reporting import (
    control_loop_fields,
)

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
            if self._features.is_on(_STAGE_FEATURES[stage[0]])
            and self._has_collaborator(stage[0])
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
        graph.add_edge("finalize", END)
        return graph.compile()

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
        """Act on requested tools, or finish."""
        last = self._current_messages(state)[-1] if self._current_messages(state) else None
        return "act" if last is not None and last.tool_calls else "finalize"

    def _after_act(self, state: AgentState) -> str:
        """Continue only while autonomous execution remains permitted."""
        # A Gatekeeper ESCALATE is a terminal autonomous decision for this turn.
        # Do not send the model back to reason(), or it could propose another
        # side effect after the system has explicitly handed control to a human.
        if state.escalated:
            return "finalize"
        # Some refusals are terminal for the same reason: no re-proposal
        # with different arguments can turn them into an allow. Sending the
        # model back to try again buys a slower version of the same answer,
        # which is how one refused refund became three and then the cap.
        if state.metadata.get("refusal_is_final"):
            return "finalize"
        next_round = state.tool_rounds + 1
        if next_round <= self._max_rounds:
            return "reason"
        report = self._control_loop.evaluate_round(next_round, self._max_rounds)
        self._reporter_blocked(
            "control_loop",
            summary="Maximum tool iterations reached",
            event_kind="control_loop",
            fields=control_loop_fields(report),
        )
        return "finalize"
