"""What every node group needs: state notes, halts, and the status feed.

Each node group is its own object, built from the agent's `NodeKit`. `NodeSupport` binds
the collaborators its shared helpers read; each group's constructor binds the rest of what
its own nodes read, so a group's dependencies are written where the group is.
"""

from typing import Any

from agent_platform.context.types import Chunk
from agent_platform.core.primitives.money import Money
from agent_platform.llm.types import CompletionRequest, CompletionResult, Message
from agent_platform.loop_control import LoopGovernor, LoopState, LoopVerdict
from agent_platform.memory import WorkingMemoryStore
from agent_platform.memory.procedural.skills import Activation, AgentInstructions
from agent_platform.observability import (
    StageReporter,
)
from agent_platform.observability.loop_reporter import mark_step
from agent_platform.observability.step_events import LOOP_STAGES, StepEvents
from agent_platform.observability.types import EventKind
from agent_platform.security.auth.policy import AgentPolicy
from agent_platform.security.pii.active import tokenise_if_on
from agent_platform.tools import NamespacedToolRegistry
from customer_support_agent.config.settings import get_agent_settings
from customer_support_agent.domain.order_reference import (
    ORDER_UNDER_DISCUSSION,
    OrderReferenceResolver,
)
from customer_support_agent.gatekeeper.business_rules import REFUND_ACTION_CLASS
from customer_support_agent.graph.kit import NodeKit
from customer_support_agent.graph.state import AgentState
from customer_support_agent.grounding.evidence import (
    GroundingEvidence,
    TurnRecord,
    order_under_discussion,
    turn_actions,
)
from customer_support_agent.memory.signals import turn_signals

# The reads a skill ran before the first model call (model names), for its card.
PREFETCHED = "prefetched"
# The tools this turn offers the model (qualified names), settled once before any read.
TURN_OFFERED_TOOLS = "turn_offered_tools"
# The owned order row the deterministic checks read this turn (JSON), so later
# stages answer from it instead of reading the same order again.
ORDER_SNAPSHOT = "order_snapshot"


# The route a turn takes when no intent was placed (`intent_execution_route`).
_UNPLACED_ROUTE = "llm_fallback"


class NodeSupport:
    """The helpers every node group shares, and the collaborators those helpers read.

    Each group is its own object, built from the agent's `NodeKit`; it binds what its own
    nodes read on top of these. A group that needs another group's work (Execution's
    `invoke`, Proposal's prompt) is handed that group, so the coupling is a constructor
    argument rather than a method found on a shared object.
    """

    def __init__(self, kit: NodeKit) -> None:
        self._agent_config = get_agent_settings()
        self._registry: NamespacedToolRegistry = kit.registry
        self._working_memory: WorkingMemoryStore | None = kit.working_memory
        # The profile, plus the skills a turn's signals select (ADR 0010).
        self._instructions: AgentInstructions = kit.instructions
        self._reporter: StageReporter | None = kit.reporter
        self._agent_policy: AgentPolicy = kit.agent_policy
        # Every tier a reply is grounded on, assembled per turn (`grounding/evidence.py`).
        self._evidence: GroundingEvidence = kit.evidence
        self._order_reference: OrderReferenceResolver = kit.order_reference
        self._max_rounds: int = kit.max_rounds
        # Which tool-use loop iteration is running (None outside the loop).
        self._steps: StepEvents = kit.steps
        self._governor: LoopGovernor = kit.governor
        self._logger: Any = kit.logger

    def _refund_limit(self) -> Money | None:
        """The per-refund limit the prompt states: the DB row, read when the agent was built."""
        return self._agent_policy.per_action_limit(REFUND_ACTION_CLASS)

    def _resolve_read_order_id(self, state: AgentState) -> int | None:
        """Which order this request is about, if any.

        Shared rather than per-node: the read path and the write path
        disagreeing about which order is under discussion is how an agent
        answers about one order and acts on another.
        """
        return self._order_reference.resolve(
            state.query, state.metadata, state.working_memory, customer_id=state.customer_id
        )

    def _carry_order_reference(self, state: AgentState) -> dict[str, Any]:
        """Settle the order under discussion for every stage after this one.

        Resolved once, early, and carried in the turn's metadata, because
        three components need the same answer: DeterministicCheck, which will
        not guess an identifier; retrieval, which otherwise ranks eleven
        near-identical orders against a query that names none; and the
        mandate issuer, which must know what the customer asked about
        before it can say whether they asked for this.
        """
        order_id = self._resolve_read_order_id(state)
        if order_id is None:
            return {}
        return {"metadata": {**state.metadata, ORDER_UNDER_DISCUSSION: order_id}}

    def history_for(self, session_id: str) -> list[Message]:
        """The bounded conversation history from common working memory."""
        if self._working_memory is None:
            return []
        history = self._working_memory.get(session_id).history
        return [Message(**item) if isinstance(item, dict) else item for item in history]

    def _skill_signals(self, state: AgentState) -> frozenset[str]:
        """What this turn tells the skill selector: its intent and its route.

        Which metadata keys count, and how each is written, is configuration;
        a key the turn did not set contributes nothing.
        """
        return turn_signals(state.metadata)

    def _activation(self, state: AgentState) -> Activation:
        """This turn's skills — deterministic, so every stage that asks gets the same."""
        activation = self._instructions.select(
            self._skill_signals(state), state.model_query or state.query
        )
        if state.metadata.get("intent_execution_route") != _UNPLACED_ROUTE:
            return activation
        # An unplaced turn is never given a skill that writes: its words may resemble one
        # ("refund"), but a write needs a request the router recognised as one.
        reads = tuple(item for item in activation.skills if not self._writes(item.skill))
        return activation.model_copy(update={"skills": reads})

    def _writes(self, skill: Any) -> bool:
        """Whether a skill uses any tool that changes something."""
        tools = (self._registry.get(name) for name in skill.document.tools)
        return any(tool is not None and tool.side_effecting for tool in tools)

    def _skill_tools(self, state: AgentState) -> set[str]:
        """The tools this turn's skills use, by internal name — on every path.

        An unplaced turn is narrowed too: the skills chosen for it are its best guess
        at the task, and their tools are what that task needs. Empty — no narrowing — only
        when no skill is active (procedural memory withheld, or nothing matched).
        """
        names = self._activation(state).tools()
        return {tool.name for name in names if (tool := self._registry.get(name)) is not None}

    def _current_messages(self, state: AgentState) -> list[Message]:
        """Return message-shaped entries from the current working-memory turn."""
        return [
            item for item in state.working_memory.current_messages() if isinstance(item, Message)
        ]

    def _turn_record(self, state: AgentState) -> TurnRecord:
        """The turn as the grounding evidence reads it: plain values, no graph state."""
        return TurnRecord(
            query=state.query,
            asked=state.model_query or state.query,
            customer_id=state.customer_id,
            metadata=state.metadata,
            actions=turn_actions(state.working_memory.current_turn),
        )

    def _grounding_evidence(self, state: AgentState) -> list[Chunk]:
        """Every tier a reply may be grounded on, each labelled with its tier."""
        return self._evidence.for_turn(self._turn_record(state))

    def _record_evidence(self, state: AgentState) -> list[Chunk]:
        """The order record, then this turn's tool results: what the turn actually read."""
        return self._evidence.records(self._turn_record(state))

    def _authoritative_observations(self, state: AgentState) -> list[str]:
        """What this turn's tools established — including what they refused."""
        return self._evidence.authoritative(self._turn_record(state))

    def _evidence_sources(self, state: AgentState) -> list[str]:
        """Where this turn's record evidence came from: the order record, then each tool."""
        return self._evidence.sources(self._turn_record(state))

    def _order_record(self, state: AgentState) -> Chunk | None:
        """The order under discussion as the database holds it now, if one is settled."""
        return self._evidence.order_record(self._turn_record(state))

    def _observations(self, state: AgentState) -> list[tuple[str, bool, str]]:
        """Each tool observation, whether it changed anything, and its tool."""
        return self._evidence.observations(self._turn_record(state))

    def _policy_evidence(self) -> list[Chunk]:
        """Every rule the model was given, one chunk per rule, in two tiers."""
        return self._evidence.rules()

    @staticmethod
    def _order_under_discussion(state: AgentState) -> int | None:
        """The order this turn settled on, if it settled on one."""
        return order_under_discussion(state.metadata)

    def _add_action(
        self,
        state: AgentState,
        name: str,
        arguments: Any,
        result: str,
        *,
        outcome: str,
    ) -> None:
        """Record one tool action, and how it ended, in current-turn working memory."""
        if isinstance(arguments, str):
            try:
                import json

                arguments = json.loads(arguments)
            except (TypeError, ValueError):
                arguments = {"raw": arguments}
        state.working_memory.add_current_turn(
            {
                "kind": "action",
                "name": name,
                "arguments": arguments,
                "result": result,
                "source": name,
                "outcome": outcome,
            }
        )

    def _note(self, state: AgentState, key: str, value: Any, category: str = "") -> None:
        """Hold a fact this turn established in the session's window.

        Working memory is what stops the next turn re-deriving what this
        one already knows — which order the customer selected, what a
        tool returned, who they are. It is read back in
        `assemble_context`, so anything noted here is in front of the
        model on the following turn without another database read.

        Silent when no memory is wired: memory is optional in this
        graph, and an agent without it must still answer.
        """
        if self._working_memory is None or not state.session_id:
            return
        safe_value = tokenise_if_on(str(value))
        self._working_memory.get(state.session_id).set(
            key, tokenise_if_on(value) if isinstance(value, str) else value
        )
        self._done(
            "working_memory_write",
            event_kind="memory",
            summary=f"SET · {key} = {safe_value}",
            fields={"key": key, "memory_value": safe_value},
            quiet=True,
        )

    def _halt(
        self,
        state: AgentState,
        response: str,
        decision: str,
        *,
        remembered_as: str | None = None,
    ) -> dict[str, Any]:
        """Answer now, without reaching the model.

        `remembered_as` is what this answer looks like in working memory,
        for the answers that are much larger on screen than they are in
        substance. A rendered table of every order the customer has
        placed is the right reply to "show me my orders" and the wrong
        thing to replay into every prompt for the rest of the session;
        the customer still sees the table, and the model reads the line.
        """
        return {
            "response": response,
            "decisions": [*state.decisions, decision],
            "metadata": {
                **state.metadata,
                "halted": "answered_from_db",
                **({} if remembered_as is None else {"memory_response": remembered_as}),
            },
        }

    def _report_llm_request(self, request: CompletionRequest, *, purpose: str) -> None:
        """LLM REQUEST: the platform's card for what the model is handed."""
        self._steps.llm_request(request, purpose=purpose)

    def _report_llm_response(self, result: CompletionResult, *, purpose: str) -> None:
        """LLM RESPONSE: the platform's card for what came back."""
        self._steps.llm_response(result, purpose=purpose)

    def _report(
        self,
        stage: str,
        detail: str | None = None,
        *,
        event_kind: EventKind = "stage",
        summary: str | None = None,
        fields: dict[str, Any] | None = None,
    ) -> None:
        """Emit a stage-started event for the TUI, when one is listening."""
        if self._reporter is not None:
            summary, fields = self._stepped(stage, summary, fields)
            self._reporter.started(
                stage,
                detail=detail,
                event_kind=event_kind,
                summary=summary,
                fields=fields,
            )

    def _done(
        self,
        stage: str,
        tokens: int | None = None,
        detail: str | None = None,
        *,
        event_kind: EventKind = "stage",
        summary: str | None = None,
        fields: dict[str, Any] | None = None,
        quiet: bool = False,
    ) -> None:
        """Emit a stage-completed event with safe semantic fields.

        `quiet` events are logged and counted in the turn summary without
        drawing a card of their own.
        """
        if self._reporter is not None:
            summary, fields = self._stepped(stage, summary, fields)
            self._reporter.completed(
                stage,
                tokens_delta=tokens,
                detail=detail,
                event_kind=event_kind,
                summary=summary,
                fields=fields,
                quiet=quiet,
            )

    def _done_failed(
        self,
        stage: str,
        detail: str | None = None,
        *,
        event_kind: EventKind = "stage",
        summary: str | None = None,
        fields: dict[str, Any] | None = None,
    ) -> None:
        """Emit a stage-failed event so the feed shows what broke."""
        if self._reporter is not None:
            summary, fields = self._stepped(stage, summary, fields)
            self._reporter.failed(
                stage,
                detail=detail,
                event_kind=event_kind,
                summary=summary,
                fields=fields,
            )

    def _report_loop_status(
        self,
        verdict: LoopVerdict,
        state: AgentState,
        *,
        tools_requested: bool,
        next_step: str,
        because: str,
        escalated: bool | None = None,
        refusal_final: bool | None = None,
    ) -> None:
        """Show every criterion the loop weighs, its value now, and the step it chose.

        Reported by the node that just ran (on the event loop, where a card
        can be drawn), from the same decision the graph routes on. Reported
        every round, not only when a criterion fires: a card that appears
        only on failure leaves a passing round unexplained.
        """
        fields = self._steps.loop_verdict(
            verdict,
            criteria={
                "model_requested_tools": tools_requested,
                "duplicate_action": duplicate_suppressed(self._governor.state),
                "escalated_to_human": state.escalated if escalated is None else escalated,
                "final_refusal": (
                    bool(state.metadata.get("refusal_is_final"))
                    if refusal_final is None
                    else refusal_final
                ),
            },
            next_step=next_step,
            because=because,
            because_words=self._agent_config.loop_control_status.because,
        )
        self._logger.info("Loop control status", **fields)

    def _stepped(
        self, stage: str, summary: str | None, fields: dict[str, Any] | None
    ) -> tuple[str | None, dict[str, Any] | None]:
        """Mark a card as part of tool-use loop step N (model call → tool → result).

        The loop is typed tool use, not ReAct: the model proposes a
        validated call rather than writing a thought, so the label says
        what happens — a model call, the tool it asked for, its result —
        and nothing that implies a thought step. Numbering the iteration
        is what makes "the model asked twice" visible. The wording is
        configuration, so it cannot drift from what the loop does.
        """
        return mark_step(
            stage,
            summary,
            fields,
            step=self._steps.step,
            max_steps=self._max_rounds,
            stages=LOOP_STAGES,
        )

    def _reporter_blocked(
        self,
        stage: str,
        *,
        detail: str | None = None,
        summary: str | None = None,
        event_kind: EventKind = "decision",
        fields: dict[str, Any] | None = None,
    ) -> None:
        if self._reporter is not None:
            summary, fields = self._stepped(stage, summary, fields)
            self._reporter.blocked(
                stage,
                detail=detail,
                event_kind=event_kind,
                summary=summary,
                fields=fields,
            )

    def _reporter_warning(
        self,
        stage: str,
        *,
        detail: str | None = None,
        summary: str | None = None,
        event_kind: EventKind = "system",
        fields: dict[str, Any] | None = None,
    ) -> None:
        if self._reporter is not None:
            summary, fields = self._stepped(stage, summary, fields)
            self._reporter.warning(
                stage,
                detail=detail,
                event_kind=event_kind,
                summary=summary,
                fields=fields,
            )

    def _reporter_waiting(
        self,
        stage: str,
        *,
        detail: str | None = None,
        summary: str | None = None,
        event_kind: EventKind = "stage",
        fields: dict[str, Any] | None = None,
    ) -> None:
        if self._reporter is not None:
            summary, fields = self._stepped(stage, summary, fields)
            self._reporter.waiting(
                stage,
                detail=detail,
                event_kind=event_kind,
                summary=summary,
                fields=fields,
            )


def duplicate_suppressed(loop: LoopState) -> bool:
    """Whether this turn's governor suppressed a repeated action."""
    return any(record.suppressed for record in loop.actions)
