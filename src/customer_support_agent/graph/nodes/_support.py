"""What every node group needs: state notes, halts, and the status feed.

The attribute declarations below are the interesting part. Each node
group is a mixin over the one agent, so without them there is nothing
saying which collaborators a group actually uses — and a type checker
could not tell you either. Declared here once, they are a readable
inventory of what the graph is composed of, and `mypy --strict` holds
every node group to it.
"""

import re
from functools import cache
from typing import Any

from agent_platform.audit import AuditLedger
from agent_platform.config import FeatureSet, get_settings
from agent_platform.context import (
    ContextFailureDetector,
    FiveStagePipeline,
    StrategySelector,
)
from agent_platform.context.types import Chunk
from agent_platform.deterministic_checks import DeterministicChecker
from agent_platform.durability import DurableGraphRunner, RunRegistry, SqliteCheckpointer
from agent_platform.evals import NollmScoringCascadeImpl
from agent_platform.executor import ToolExecutor
from agent_platform.gatekeeper import Gatekeeper
from agent_platform.gatekeeper.mandate import MandateIssuer
from agent_platform.guardrails import ResponseGuardrail
from agent_platform.guardrails.response import NOT_EXECUTED_MARKER
from agent_platform.hallucination_check import (
    DetectionLadder,
    GroundingReasoningDiagnoser,
    Observation,
    tool_evidence,
)
from agent_platform.llm.client import LLMClient
from agent_platform.llm.types import CompletionRequest, CompletionResult, Message
from agent_platform.loop_control import LoopGovernor, LoopState, LoopVerdict
from agent_platform.memory import MemoryFacade, WorkingMemoryStore
from agent_platform.memory.semantic.rag import KNOWLEDGE_SOURCE_PREFIX, section_sentences
from agent_platform.observability import (
    StageReporter,
    Tracer,
)
from agent_platform.observability.loop_reporter import mark_step
from agent_platform.observability.step_events import LOOP_STAGES, StepEvents
from agent_platform.observability.types import EventKind
from agent_platform.primitives.money import Money, money_or_none
from agent_platform.resilience import (
    DeadLetterQueue,
)
from agent_platform.security import MessageScreener, SecuritySuite
from agent_platform.security.auth.policy import AgentPolicy
from agent_platform.skills import Activation, AgentInstructions
from agent_platform.tools import NamespacedToolRegistry, ToolRetrievalSelector
from agent_platform.tools.path import ToolPath
from agent_platform.verification import EffectVerifier
from customer_support_agent.config.settings import get_agent_settings
from customer_support_agent.domain.identity import (
    Customer,
    current_customer,
)
from customer_support_agent.domain.models import Order
from customer_support_agent.domain.order_reference import (
    ORDER_UNDER_DISCUSSION,
    OrderReferenceResolver,
)
from customer_support_agent.gatekeeper.business_rules import REFUND_ACTION_CLASS
from customer_support_agent.graph.state import AgentState
from customer_support_agent.intent.router import CustomerSupportIntentRouter, Intent
from customer_support_agent.loop_control import OutputQualityGate
from customer_support_agent.observability.accounting import TurnAccountant
from customer_support_agent.persistence.repositories import OrdersRepository
from customer_support_agent.presentation.facts import order_facts, policy_facts, session_facts
from customer_support_agent.presentation.orders import (
    OrderMenu,
    order_record,
)
from customer_support_agent.prompts.policy import policy_text
from customer_support_agent.prompts.system_prompt import (
    build_reply_instruction,
    build_reply_system_prompt,
    build_system_prompt,
    refund_limit_notice,
)
from customer_support_agent.review.referral import ReplyReferral, ReviewReferral
from customer_support_agent.security.authentication import CustomerAuthenticator
from customer_support_agent.security.limit_disclosure import LimitDisclosureCheck
from customer_support_agent.tools.definitions import SupportTools

# How a tool call ended, as working memory records it and grounding reads
# it. Only a completed call shows its effect happened; a referred call
# shows a person will decide, which the referral really did make true;
# anything else shows only that the call was refused.
TOOL_COMPLETED = "completed"
TOOL_REFERRED = "referred"
TOOL_BLOCKED = "blocked"
# The reads a procedure ran before the first model call (model names), for its card.
PREFETCHED = "prefetched"
# The tools this turn offers the model (qualified names), settled once before any read.
TURN_OFFERED_TOOLS = "turn_offered_tools"
# The owned order row the deterministic checks read this turn (JSON), so later
# stages answer from it instead of reading the same order again.
ORDER_SNAPSHOT = "order_snapshot"
# The outcomes whose recorded result is itself the evidence.
_GROUNDED_AS_STATED = frozenset({TOOL_COMPLETED, TOOL_REFERRED})


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
    # The profile, plus the procedures a turn's signals select (ADR 0010).
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
    _features: FeatureSet
    _intents: CustomerSupportIntentRouter
    _authenticator: CustomerAuthenticator | None
    _orders: OrdersRepository | None
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
        """Run the reads this turn's procedures declare come first."""
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

    def _procedure_signals(self, state: AgentState) -> frozenset[str]:
        """What this turn tells the procedure selector: its intent and its route.

        Which metadata keys count, and how each is written, is configuration;
        a key the turn did not set contributes nothing.
        """
        templates = get_agent_settings().instruction_signals
        return frozenset(
            template.format(value=state.metadata[key])
            for key, template in templates.items()
            if state.metadata.get(key)
        )

    def _activation(self, state: AgentState) -> Activation:
        """This turn's procedures — deterministic, so every stage that asks gets the same."""
        return self._instructions.select(
            self._procedure_signals(state), state.model_query or state.query
        )

    def _procedure_tools(self, state: AgentState) -> set[str]:
        """The tools this turn's procedures use, by internal name — on every path.

        An unplaced turn is narrowed too: the procedures chosen for it are its best guess
        at the task, and their tools are what that task needs. Empty — no narrowing — only
        when no procedure is active (procedural memory withheld, or nothing matched).
        """
        names = self._activation(state).tools()
        return {tool.name for name in names if (tool := self._registry.get(name)) is not None}

    def _current_messages(self, state: AgentState) -> list[Message]:
        """Return message-shaped entries from the current working-memory turn."""
        return [
            item for item in state.working_memory.current_messages() if isinstance(item, Message)
        ]

    def _authoritative_observations(self, state: AgentState) -> list[str]:
        """What this turn's tools established — including what they refused.

        A refusal is evidence too: "your $0.01 refund was rejected" is a
        true statement the reply must be allowed to make. Refused calls
        are rendered as *not executed*, with the amount that was asked
        for, so they ground the refusal and nothing else — a claim that
        money moved still needs a settled state from a successful call.

        A call the harness referred for review is grounded on what the
        referral did — queued it, and left the refund pending — not as a
        refusal, which NLI reads as contradicting "your request is in".
        """
        record = self._order_record(state)
        return [
            *([record.content] if record is not None else []),
            *(text for text, _, _ in self._observations(state)),
        ]

    def _order_record(self, state: AgentState) -> Chunk | None:
        """The order under discussion as the database holds it now, if one is settled.

        The model is shown this order in its context window; the reply it
        writes from it is checked against the same row, read again here
        rather than taken back out of the window. Only that one order: a
        refund or cancellation is never about any other.
        """
        order_id = self._order_under_discussion(state)
        if order_id is None or self._orders is None:
            return None
        return order_record(self._orders, order_id, state.customer_id)

    @staticmethod
    def _order_under_discussion(state: AgentState) -> int | None:
        """The order this turn settled on, if it settled on one."""
        order_id = state.metadata.get(ORDER_UNDER_DISCUSSION)
        if isinstance(order_id, bool) or not isinstance(order_id, int):
            return None
        return order_id

    def _evidence_sources(self, state: AgentState) -> list[str]:
        """Where this turn's record evidence came from: the order record, then each tool."""
        record = self._order_record(state)
        return [
            *([record.source] if record is not None else []),
            *(tool for _, _, tool in self._observations(state)),
        ]

    def _observations(self, state: AgentState) -> list[tuple[str, bool, str]]:
        """Each of this turn's tool observations, whether it changed anything, and its tool.

        A call changed something when its tool is declared side-effecting
        *and* it ran (or was referred): a refused write left the world as
        it was, so the reads before it stay true. The grounding guard needs
        the distinction to tell a read that a later write has overtaken.
        """
        actions = [
            item
            for item in state.working_memory.current_turn
            if isinstance(item, dict) and item.get("kind") == "action"
        ]
        return [
            (
                str(item.get("result", ""))
                if item.get("outcome") in _GROUNDED_AS_STATED
                else _not_executed(item),
                item.get("outcome") in _GROUNDED_AS_STATED and self._changes_state(item),
                # The tool's own name (`support__get_order_status` -> `get_order_status`), which
                # the HALLUCINATION CHECK card names as the line's source.
                str(item.get("name", "")).rsplit("__", maxsplit=1)[-1],
            )
            for item in actions
        ]

    def _changes_state(self, action: dict[str, Any]) -> bool:
        """Whether the tool this action called is declared to have side effects."""
        tool = self._registry.get(str(action.get("name", "")))
        return tool is not None and tool.side_effecting

    def _grounding_evidence(self, state: AgentState) -> list[Chunk]:
        """Every tier a reply may be grounded on, each labelled with its tier.

        Tool results (the only tier that can show something happened), the
        refund policy the model was given, and the knowledge and verified
        memories the context node let into the window. Which tier may
        support which kind of claim is the guardrail's rule, not this one's.
        """
        # In the order they happened: a read taken before a refund was queued
        # is superseded by the queueing, and the checker needs to know which
        # came last to let it. A multi-line result (a list of orders) is one
        # chunk per line: the checker compares a claim against one premise at
        # a time, and a whole table is a premise NLI cannot read.
        #
        # And one chunk per sentence, for the same reason: "Order 10693 has
        # been cancelled. Order 10693 is already awaiting human review. …"
        # read as *neutral* on "your order has been cancelled", and an earlier
        # read of the still-pending order was left to contradict it. A refusal
        # stays whole, so its marker is never parted from what was refused.
        # What the customer said, as the model saw it. A reply may echo it
        # ("I can't issue a $1 refund") but it proves nothing, so its tier
        # is configured support-only.
        asked = [
            Chunk(
                content=state.model_query or state.query, kind="history", source="customer_message"
            )
        ]
        # Knowledge passages the window retrieved are already covered: the
        # whole policy corpus is evidence below, one section per chunk.
        seen = [
            Chunk(content=item["content"], kind=item["kind"], source=item["source"])
            for item in state.metadata.get(GROUNDING_EVIDENCE) or []
            if not str(item.get("source", "")).startswith(KNOWLEDGE_SOURCE_PREFIX)
        ]
        return [
            *self._record_evidence(state),
            *self._policy_evidence(),
            *_knowledge_evidence(),
            *self._typed_facts(state),
            *asked,
            *seen,
        ]

    def _record_evidence(self, state: AgentState) -> list[Chunk]:
        """The order record, then this turn's tool results: what the turn actually read.

        The order record goes first: it was read for the turn, and anything
        this turn's tools wrote after it supersedes it in the guard's order.
        """
        record = self._order_record(state)
        tools = tool_evidence([Observation(*observed) for observed in self._observations(state)])
        return [*([record] if record is not None else []), *tools]

    def _typed_facts(self, state: AgentState) -> list[Chunk]:
        """The order, the rules and the session as typed facts the guard reads first.

        Who the customer is comes from authentication (the customer cannot set it),
        so "You're Dennis Johnson" is grounded by the session, not by their say-so.
        The order is read as it stands now: a write this turn evicted the cached row.
        """
        order = self._discussed_order(state)
        return [
            *(order_facts(order) if order is not None else []),
            *policy_facts(self._refund_limit()),
            *session_facts(current_customer()),
        ]

    def _discussed_order(self, state: AgentState) -> Order | None:
        """The order under discussion, if this customer owns it."""
        order_id = self._order_under_discussion(state)
        if order_id is None or self._orders is None or state.customer_id is None:
            return None
        return self._orders.get_for_customer(order_id, state.customer_id)

    def _policy_evidence(self) -> list[Chunk]:
        """Every rule the model was given, one chunk per rule, in two tiers.

        The refund policy file is `policy`: the business's rules, which may
        support or refute a claim. The rest of the system prompt is
        `system`: instructions to the model, which may support a reply that
        restates them — the politeness it was told to use, the rule it was
        told to cite — but are never facts a correct refusal can contradict.

        One rule per chunk, not the whole prompt: a claim is judged against
        the rule it is about, and a paragraph of unrelated rules (with their
        "do not"s) cannot read as contradicting it.
        """
        policy = _prompt_rules(policy_text())
        limit = self._refund_limit()
        if limit:
            policy.append(refund_limit_notice(limit).strip())
        # Every prompt a reply is written under, not only the one that chose
        # the tools: the final call is made under the reply prompt, and the
        # manner it asks for is what a courteous sentence restates.
        given = _prompt_rules(
            "\n\n".join(
                (
                    build_system_prompt(refund_limit=limit),
                    build_reply_system_prompt(),
                    build_reply_instruction(),
                )
            )
        )
        instructions = list(dict.fromkeys(rule for rule in given if rule not in policy))
        return [
            *(
                Chunk(content=rule, kind="policy", source=f"refund_policy#{index}")
                for index, rule in enumerate(policy, start=1)
            ),
            *(
                Chunk(content=rule, kind="system", source=f"system_prompt#{index}")
                for index, rule in enumerate(instructions, start=1)
            ),
        ]

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
        self._working_memory.get(state.session_id).set(key, value)
        safe_value = self._security.redact(str(value))
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


# Where the context node leaves the window's grounding sources for finalize.
GROUNDING_EVIDENCE = "grounding_evidence"


# A sentence boundary inside a rule: terminal punctuation, then a capital.
_RULE_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _prompt_rules(markdown: str) -> list[str]:
    """Each bullet, and each plain paragraph, of a prompt as one rule.

    A bullet's continuation lines belong to it; a blank line ends a
    paragraph. Headings are structure, not rules.
    """
    rules: list[str] = []
    open_rule = False
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            open_rule = False
        elif stripped.startswith("- ") or not open_rule:
            rules.append(stripped.removeprefix("- "))
            open_rule = True
        else:
            rules[-1] = f"{rules[-1]} {stripped}"
    # One sentence per rule. A bullet that states a threshold, forbids a
    # tool and scripts the reply is three rules; embedded whole, it is
    # retrieved for none of the claims that restate just one of them.
    return [sentence for rule in rules for sentence in _RULE_SENTENCE.split(rule) if sentence]


@cache
def _knowledge_evidence() -> tuple[Chunk, ...]:
    """The agent's policy articles, one chunk per sentence, read once.

    The articles are the business's own statement of its policy, so they
    ground what the reply says about how refunds work — whether or not
    retrieval happened to put that section in this turn's window.

    One sentence per chunk, as `_policy_evidence` does for the policy file: a
    whole section ("… a delivered order can be refunded … an order still in
    flight has nothing to refund …") read as contradicting an offer to refund a
    delivered order, because NLI weighs the paragraph, not the rule it is about.
    """
    rag = get_settings().memory.semantic.rag
    return tuple(
        Chunk(
            content=sentence,
            kind="policy",
            source=f"{KNOWLEDGE_SOURCE_PREFIX}{path.name}#{index}.{number}",
        )
        for path in sorted(rag.resolved_corpus_dir.glob(rag.corpus_glob))
        for index, sentences in enumerate(section_sentences(path.read_text(encoding="utf-8")))
        for number, sentence in enumerate(sentences)
    )


def _not_executed(action: dict[str, Any]) -> str:
    """A refused or failed tool call as evidence of exactly that."""
    arguments = action.get("arguments")
    named = dict(arguments) if isinstance(arguments, dict) else {}
    amount = money_or_none(named.get("amount"))
    if amount is not None:
        named["amount"] = str(amount)
    shown = ", ".join(f"{key}={value}" for key, value in named.items())
    return f"{NOT_EXECUTED_MARKER} {action.get('name', '')}({shown}): {action.get('result', '')}"


def duplicate_suppressed(loop: LoopState) -> bool:
    """Whether this turn's governor suppressed a repeated action."""
    return any(record.suppressed for record in loop.actions)
