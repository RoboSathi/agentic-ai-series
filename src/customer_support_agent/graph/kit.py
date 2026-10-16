"""What the nodes work with: the parts, and the collaborators built from them once.

`CsaParts` is what the composition root supplies. Several collaborators are derived from
those parts — the one tool path, the screener, the order-reference resolver, the grounding
assembler, the loop governor, the step narrator. `NodeKit` builds them once, here, so every
node group and the agent shell are handed the same instances rather than each building its
own, and so the derivation is written in one place.
"""

from typing import Any

from agent_platform.audit import AuditLedger, NullAuditLedger
from agent_platform.config import FeatureSet, get_settings
from agent_platform.context import ContextFailureDetector, FiveStagePipeline, StrategySelector
from agent_platform.deterministic_checks import DeterministicChecker
from agent_platform.durability import RunRegistry, SqliteCheckpointer
from agent_platform.gatekeeper import Gatekeeper
from agent_platform.gatekeeper.mandate import MandateIssuer
from agent_platform.guardrails import ResponseGuardrail
from agent_platform.hallucination_check import DetectionLadder
from agent_platform.llm.client import LLMClient
from agent_platform.logging import get_logger
from agent_platform.loop_control import LoopBudget, LoopGovernor, build_loop_governor
from agent_platform.memory import MemoryFacade, TurnRecorder, WorkingMemoryStore
from agent_platform.memory.procedural.skills import AgentInstructions, build_agent_instructions
from agent_platform.observability import StageReporter, Tracer
from agent_platform.observability.langsmith import NullTraceExporter, TraceExporter
from agent_platform.observability.step_events import StepEvents
from agent_platform.observability.units import declare_screens
from agent_platform.resilience import DeadLetterQueue
from agent_platform.runtime.executor import ToolExecutor
from agent_platform.runtime.verification import EffectVerifier
from agent_platform.scoring import NollmScoringCascadeImpl
from agent_platform.security import MessageScreener, SecuritySuite
from agent_platform.security.auth.policy import AgentPolicy
from agent_platform.tools import NamespacedToolRegistry, ToolRetrievalSelector
from agent_platform.tools.path import ToolPath
from customer_support_agent.config import get_agent_settings
from customer_support_agent.domain.order_reference import OrderReferenceResolver
from customer_support_agent.gatekeeper.business_rules import REFUND_ACTION_CLASS
from customer_support_agent.graph.conversation import ConversationState
from customer_support_agent.graph.parts import CsaParts
from customer_support_agent.grounding.evidence import GroundingEvidence
from customer_support_agent.intent.phrases import PhraseReader
from customer_support_agent.intent.router import CustomerSupportIntentRouter
from customer_support_agent.loop_control import IndependentScorer, OutputQualityGate
from customer_support_agent.memory.turn_memory import TurnMemory
from customer_support_agent.observability.accounting import TurnAccountant
from customer_support_agent.persistence.repositories import OrdersRepository
from customer_support_agent.review.referral import ReplyReferral, ReviewReferral
from customer_support_agent.security.authentication import CustomerAuthenticator
from customer_support_agent.security.limit_disclosure import LimitDisclosureCheck


class NodeKit:
    """The agent's collaborators, supplied and derived, built once per agent."""

    def __init__(self, parts: CsaParts) -> None:
        self._app_config = get_settings()
        self._agent_config = get_agent_settings()
        self._supplied(parts)
        self._derive_tooling(parts)
        self._derive_turn(parts)

    def _supplied(self, parts: CsaParts) -> None:
        """The parts as given, under the names the nodes read them by."""
        model, tools, guards, memory, run = (
            parts.model, parts.tools, parts.guards, parts.memory, parts.run
        )  # fmt: skip
        # Which stages this agent has. Read once, here, because the graph is compiled from it —
        # a feature flipped afterwards reaches a running agent only by rebuilding one.
        self.features: FeatureSet = (
            parts.features if parts.features is not None else self._app_config.features
        )
        self.llm: LLMClient = model.llm
        self.selector: ToolRetrievalSelector | None = model.selector
        self.tracer: Tracer | None = model.tracer
        self.cascade: NollmScoringCascadeImpl | None = model.cascade
        self.gatekeeper: Gatekeeper | None = tools.gatekeeper
        # Which refund-limit sentence the prompt carries (`refund_limit_notice`).
        self.governed = tools.gatekeeper is not None
        # Written by the Gatekeeper when there is one, and by the agent itself when there is
        # not — an ungoverned execution is exactly the one the trail must not be missing.
        self.audit: AuditLedger = tools.audit or NullAuditLedger()
        self.registry: NamespacedToolRegistry = tools.registry
        self.executor: ToolExecutor = tools.executor
        self.verifier: EffectVerifier | None = tools.verifier
        self.mandate_issuer: MandateIssuer | None = tools.mandate_issuer
        self.referral: ReviewReferral | None = tools.referral
        self.reply_referral: ReplyReferral | None = tools.reply_referral
        self.security: SecuritySuite = guards.security
        self.agent_policy: AgentPolicy = guards.agent_policy
        self.intents: CustomerSupportIntentRouter = guards.intent_router
        self.authenticator: CustomerAuthenticator | None = guards.authenticator
        self.orders: OrdersRepository | None = guards.orders
        self.deterministic_checks: DeterministicChecker | None = guards.deterministic_checks
        self.grounding: DetectionLadder | None = guards.grounding
        self.dead_letter_queue: DeadLetterQueue | None = guards.dead_letter_queue
        self.working_memory: WorkingMemoryStore | None = memory.working_memory
        self.long_term: MemoryFacade | None = memory.long_term_memory
        self.dream_recorder: TurnRecorder | None = memory.dream_recorder
        self.context: FiveStagePipeline | None = memory.context
        self.context_health: ContextFailureDetector | None = memory.context_health
        self.strategies: StrategySelector | None = memory.context_strategies
        self.checkpointer: SqliteCheckpointer | None = run.checkpointer
        self.runs: RunRegistry | None = run.runs
        self.reporter: StageReporter | None = run.reporter
        # Where each turn's trace is exported (LangSmith), if anywhere.
        self.trace_exporter: TraceExporter = run.trace_exporter or NullTraceExporter()
        # Eight SLOs are configured; this is what actually observes them, once per turn.
        self.accountant = TurnAccountant(slos=run.slos)
        self.conversation = ConversationState()
        self.logger: Any = get_logger("customer_support_agent")

    def _derive_tooling(self, parts: CsaParts) -> None:
        """The one tool path, the screener, and the profile the prompt is built from."""
        # Every model-proposed call takes this one path (ADR 0009 D9): it also writes the
        # UNGOVERNED audit entry when the Gatekeeper is withheld.
        self.tool_path = ToolPath(self.registry, self.executor, self.audit, self.gatekeeper)
        # One screener for the agent's life; the screens it runs are declared for analytics.
        self.message_screener = MessageScreener(self.security, self.dead_letter_queue)
        declare_screens(self._app_config.agent.id, self.message_screener.declared_checks())
        # The profile always; skills only when the composition root built procedural memory.
        self.instructions: AgentInstructions = parts.model.instructions or (
            build_agent_instructions(self._app_config.skills, store=None, audit=None, reporter=None)
        )
        # What a finished turn leaves in memory: the transcript, the dream inbox, a claim.
        self.turn_memory = TurnMemory(
            self.working_memory,
            self.long_term,
            self.dream_recorder,
            self.security.redact,
            self.reporter,
        )

    def _derive_turn(self, parts: CsaParts) -> None:
        """What reads a turn: its order, its evidence, its loop, its reply checks."""
        settings = self._agent_config
        self.deliberation_markers = [m.casefold() for m in settings.deliberation_markers]
        self.max_rounds = settings.max_tool_iterations
        self.actor = self.agent_policy.agent_id
        # Token confidence (step 1) draws its card on this agent's feed.
        self.response_guardrail = ResponseGuardrail(self.grounding)
        self.limit_disclosure = LimitDisclosureCheck(
            self.agent_policy.per_action_limit(REFUND_ACTION_CLASS)
        )
        # Which order the customer is talking about, resolved once a turn from what they said
        # and what the record holds. One resolver, so the path that answers and the path that
        # acts cannot drift.
        rules, negated_at = settings.order_reference, PhraseReader().negated_at
        self.order_reference = (
            OrderReferenceResolver.from_orders(rules, self.orders, negated_at)
            if self.orders is not None
            else OrderReferenceResolver(rules, negated_at=negated_at)
        )
        # What a reply is grounded on: the order record, tools, rules, facts — one assembler.
        self.evidence = GroundingEvidence(
            self.registry, self.agent_policy, self.orders, self.order_reference,
            governed=self.governed, knowledge_window=settings.knowledge_passage_window,
        )  # fmt: skip
        # The one authority over the tool loop: step cap, deadline, budgets, repeats (exact,
        # canonical, semantic), cycles and convergence. Its state rides in the graph state.
        self.governor: LoopGovernor = parts.model.governor or build_loop_governor(
            LoopBudget(max_steps=self.max_rounds)
        )
        # Plan, reflect, evaluate. Injected as one composition so a deployment can swap any
        # of the three; deterministic by default, so deliberating costs no completions.
        self.deliberation: OutputQualityGate = parts.model.deliberation or OutputQualityGate(
            evaluator=IndependentScorer(self.cascade) if self.cascade is not None else None
        )
        self.steps = StepEvents(self.reporter, max_steps=self.max_rounds, model=self.llm.model)


__all__ = ["NodeKit"]
