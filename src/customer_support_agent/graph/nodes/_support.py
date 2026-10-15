"""What every node group needs: state notes, halts, and the status feed.

The attribute declarations below are the interesting part. Each node
group is a mixin over the one agent, so without them there is nothing
saying which collaborators a group actually uses — and a type checker
could not tell you either. Declared here once, they are a readable
inventory of what the graph is composed of, and `mypy --strict` holds
every node group to it.
"""

from typing import Any

from agent_platform.audit import AuditLedger
from agent_platform.config import FeatureSet
from agent_platform.context import (
    ContextFailureDetector,
    FiveStagePipeline,
    StrategySelector,
)
from agent_platform.context.types import Chunk
from agent_platform.core.primitives.money import Money
from agent_platform.deterministic_checks import DeterministicChecker
from agent_platform.durability import DurableGraphRunner, RunRegistry, SqliteCheckpointer
from agent_platform.gatekeeper import Gatekeeper
from agent_platform.gatekeeper.mandate import MandateIssuer
from agent_platform.guardrails import ResponseGuardrail
from agent_platform.hallucination_check import (
    DetectionLadder,
    GroundingReasoningDiagnoser,
)
from agent_platform.llm.client import LLMClient
from agent_platform.llm.types import CompletionRequest, CompletionResult, Message
from agent_platform.loop_control import LoopGovernor, LoopState, LoopVerdict
from agent_platform.memory import MemoryFacade, WorkingMemoryStore
from agent_platform.memory.procedural.skills import Activation, AgentInstructions
from agent_platform.observability import (
    StageReporter,
    Tracer,
)
from agent_platform.observability.loop_reporter import mark_step
from agent_platform.observability.step_events import LOOP_STAGES, StepEvents
from agent_platform.observability.types import EventKind
from agent_platform.resilience import (
    DeadLetterQueue,
)
from agent_platform.runtime.executor import ToolExecutor
from agent_platform.runtime.verification import EffectVerifier
from agent_platform.scoring import NollmScoringCascadeImpl
from agent_platform.security import MessageScreener, SecuritySuite
from agent_platform.security.auth.policy import AgentPolicy
from agent_platform.tools import NamespacedToolRegistry, ToolRetrievalSelector
from agent_platform.tools.path import ToolPath
from customer_support_agent.config.settings import get_agent_settings
from customer_support_agent.domain.identity import (
    Customer,
)
from customer_support_agent.domain.order_reference import (
    ORDER_UNDER_DISCUSSION,
    OrderReferenceResolver,
)
from customer_support_agent.gatekeeper.business_rules import REFUND_ACTION_CLASS
from customer_support_agent.graph.state import AgentState
from customer_support_agent.grounding.evidence import (
    GroundingEvidence,
    TurnRecord,
    order_under_discussion,
    turn_actions,
)
from customer_support_agent.intent.router import CustomerSupportIntentRouter, Intent
from customer_support_agent.loop_control import OutputQualityGate
from customer_support_agent.memory.signals import turn_signals
from customer_support_agent.observability.accounting import TurnAccountant
from customer_support_agent.persistence.repositories import OrdersRepository
from customer_support_agent.presentation.orders import (
    OrderMenu,
)
from customer_support_agent.review.referral import ReplyReferral, ReviewReferral
from customer_support_agent.security.authentication import CustomerAuthenticator
from customer_support_agent.security.limit_disclosure import LimitDisclosureCheck
from customer_support_agent.tools.definitions import SupportTools

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
    """Collaborators and small helpers shared by every node group."""

    # Injected by `CustomerSupportAgent.__init__`; declared here so each
    # mixin type-checks against the agent it is mixed into.
    _llm: LLMClient
    _gatekeeper: Gatekeeper | None
    _audit: AuditLedger
    _registry: NamespacedToolRegistry
    _tools: SupportTools
    _executor: ToolExecutor
    _tool_path: ToolPath
    _security: SecuritySuite
    _message_screener: MessageScreener
    _working_memory: WorkingMemoryStore | None
    _long_term: MemoryFacade | None
    _context: FiveStagePipeline | None
    # The profile, plus the skills a turn's signals select (ADR 0010).
    _instructions: AgentInstructions
    _selector: ToolRetrievalSelector | None
    _cascade: NollmScoringCascadeImpl | None
    _reporter: StageReporter | None
    _dead_letter_queue: DeadLetterQueue | None
    _tracer: Tracer | None
    _grounding: DetectionLadder | None
    _verifier: EffectVerifier | None
    _mandate_issuer: MandateIssuer | None
    _referral: ReviewReferral | None
    _reply_referral: ReplyReferral | None
    _agent_policy: AgentPolicy
    # Whether the Gatekeeper was built: it picks the prompt's refund-limit sentence.
    _governed: bool
    _features: FeatureSet
    _intents: CustomerSupportIntentRouter
    _authenticator: CustomerAuthenticator | None
    # Every tier a reply is grounded on, assembled per turn (`grounding/evidence.py`).
    _evidence: GroundingEvidence
    # Absent for an agent built without a database; a node group tests it before reading.
    _orders: OrdersRepository | None = None
    _order_reference: OrderReferenceResolver
    _deterministic_checks: DeterministicChecker | None
    _deliberation_markers: list[str]
    _max_rounds: int
    _actor: str
    _strategies: StrategySelector | None
    # Absent when `context.failure_detection` is withheld.
    _context_health: ContextFailureDetector | None
    # Which tool-use loop iteration is running (None outside the loop).
    _steps: StepEvents
    _diagnoser: GroundingReasoningDiagnoser
    _response_guardrail: ResponseGuardrail
    _limit_disclosure: LimitDisclosureCheck
    _sessions: dict[str, Customer]
    _menus: dict[str, OrderMenu]
    _last_supported_intent: dict[str, Intent]
    _governor: LoopGovernor
    _deliberation: OutputQualityGate
    _accountant: TurnAccountant
    # Durable execution. Absent only for an agent constructed by hand
    # without one — then the graph has no checkpoints and no approval wait.
    _checkpointer: SqliteCheckpointer | None
    _runs: RunRegistry | None
    _runner: DurableGraphRunner | None
    _logger: Any

    # The nodes a graph is wired from. Declared here rather than left
    # implicit because `edges.py` registers them by name and would
    # otherwise be the only place that knows the set exists — and a node
    # renamed in its own module would fail at graph-compile time rather
    # than at type-check time. Each is implemented by exactly one group.

    async def authenticate(self, state: AgentState) -> dict[str, Any]:
        """Establish who is asking, deterministically."""
        raise NotImplementedError

    async def await_approval(self, state: AgentState) -> dict[str, Any]:
        """Pause the run until a human decides a queued refund."""
        raise NotImplementedError

    async def choose_order(self, state: AgentState) -> dict[str, Any]:
        """Resolve which order the customer means."""
        raise NotImplementedError

    async def deterministic_checks(self, state: AgentState) -> dict[str, Any]:
        """Run this agent's own pre-flight checks."""
        raise NotImplementedError

    async def prompt_injection(self, state: AgentState) -> dict[str, Any]:
        """Screen the request for an injection attempt."""
        raise NotImplementedError

    async def content_moderation(self, state: AgentState) -> dict[str, Any]:
        """Screen the request against the content policy."""
        raise NotImplementedError

    async def pii_filtering(self, state: AgentState) -> dict[str, Any]:
        """Take personal data out of what the model will be sent."""
        raise NotImplementedError

    async def classify_intent(self, state: AgentState) -> dict[str, Any]:
        """Decide what is being asked, and what that permits."""
        raise NotImplementedError

    async def read_intent(self, state: AgentState) -> dict[str, Any]:
        """Answer a read-shaped intent without calling a model."""
        raise NotImplementedError

    async def assemble_context(self, state: AgentState) -> dict[str, Any]:
        """Build the window the model is handed."""
        raise NotImplementedError

    async def prefetch(self, state: AgentState) -> dict[str, Any]:
        """Run the reads this turn's skills declare come first."""
        raise NotImplementedError

    async def propose(self, state: AgentState) -> dict[str, Any]:
        """Ask the model what to do next."""
        raise NotImplementedError

    async def execute(self, state: AgentState) -> dict[str, Any]:
        """Execute what the model proposed, under governance."""
        raise NotImplementedError

    async def finalize(self, state: AgentState) -> dict[str, Any]:
        """Turn what happened into a reply a customer can read."""
        raise NotImplementedError

    # Helpers one group needs from another. Declaring them keeps the
    # coupling between groups visible and checked: a group that grows a
    # dependency on a sibling has to say so here.

    def _system_prompt(self) -> str:
        """The system prompt this turn is run under."""
        raise NotImplementedError

    def _refund_limit(self) -> Money | None:
        """The per-refund limit the prompt states: the DB row, read when the agent was built."""
        return self._agent_policy.per_action_limit(REFUND_ACTION_CLASS)

    def _prompt_request(
        self, state: AgentState, tools: list[dict[str, Any]] | None = None
    ) -> CompletionRequest:
        """The working request handed to the model this turn, with its manifest."""
        raise NotImplementedError

    async def _offered_tools(self, state: AgentState) -> list[Any]:
        """The tools this turn may offer the model (settled once, by the prefetch stage)."""
        raise NotImplementedError

    async def _call_model(self, request: CompletionRequest, state: AgentState) -> CompletionResult:
        """One model call, traced; its protection is the client's own (`create_llm_client`)."""
        raise NotImplementedError

    async def _invoke(
        self, name: str, arguments: str, state: AgentState, offered: set[str]
    ) -> tuple[str, str, str, dict[str, Any]]:
        """Run one proposed tool call through governance and execution."""
        raise NotImplementedError

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
        safe_value = self._security.redact(str(value))
        self._working_memory.get(state.session_id).set(
            key, self._security.redact(value) if isinstance(value, str) else value
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
            because_words=get_agent_settings().loop_control_status.because,
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
