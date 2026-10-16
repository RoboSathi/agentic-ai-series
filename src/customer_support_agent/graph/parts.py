"""What the refund agent is built from, grouped by what each part is for.

The composition root (`build_agent`) assembles these records and hands the agent one
`CsaParts`. A record is a concern — the model, the tools and their authority, the guards
in front of them, memory and context, durable execution — so a reader can see which
collaborators belong together, and a test can replace one concern without restating
the rest. Every record is frozen: an agent's wiring does not change while it runs; a
`/harness` toggle builds a new agent.
"""

from dataclasses import dataclass, field

from agent_platform.audit import AuditLedger
from agent_platform.config import FeatureSet
from agent_platform.context import ContextFailureDetector, FiveStagePipeline, StrategySelector
from agent_platform.deterministic_checks import DeterministicChecker
from agent_platform.durability import RunRegistry, SqliteCheckpointer
from agent_platform.gatekeeper import Gatekeeper
from agent_platform.gatekeeper.mandate import MandateIssuer
from agent_platform.hallucination_check import DetectionLadder
from agent_platform.llm.client import LLMClient
from agent_platform.loop_control import LoopGovernor
from agent_platform.memory import MemoryFacade, TurnRecorder, WorkingMemoryStore
from agent_platform.memory.procedural.skills import AgentInstructions
from agent_platform.observability import SLORegistry, StageReporter, Tracer
from agent_platform.observability.langsmith import TraceExporter
from agent_platform.resilience import DeadLetterQueue
from agent_platform.runtime.executor import ToolExecutor
from agent_platform.runtime.verification import EffectVerifier
from agent_platform.scoring import NollmScoringCascadeImpl
from agent_platform.security import SecuritySuite
from agent_platform.security.auth.policy import AgentPolicy
from agent_platform.tools import NamespacedToolRegistry, ToolRetrievalSelector
from customer_support_agent.intent.router import CustomerSupportIntentRouter
from customer_support_agent.loop_control import OutputQualityGate
from customer_support_agent.persistence.repositories import OrdersRepository
from customer_support_agent.review.referral import ReplyReferral, ReviewReferral
from customer_support_agent.security.authentication import CustomerAuthenticator


@dataclass(frozen=True)
class ModelParts:
    """The model, what it is told, and what judges and bounds its turns."""

    llm: LLMClient
    # The profile, plus the skills a turn's signals select (ADR 0010); the profile alone
    # when the composition root built no procedural memory.
    instructions: AgentInstructions | None = None
    selector: ToolRetrievalSelector | None = None
    tracer: Tracer | None = None
    # The one authority over the tool loop; built from `max_tool_iterations` when absent.
    governor: LoopGovernor | None = None
    # Plan, reflect, evaluate; deterministic by default.
    deliberation: OutputQualityGate | None = None
    cascade: NollmScoringCascadeImpl | None = None


@dataclass(frozen=True)
class ToolParts:
    """The tools, the path every proposed call takes, and the authority over it."""

    # Every tool the agent has, already registered (the tools themselves are its handlers).
    registry: NamespacedToolRegistry
    executor: ToolExecutor
    # Absent when the `gatekeeper` feature is withheld: calls then run UNGOVERNED.
    gatekeeper: Gatekeeper | None = None
    audit: AuditLedger | None = None
    # Reads an effect back after execution, so no claim rests on an action that did not land.
    verifier: EffectVerifier | None = None
    # Says what this turn's request authorizes.
    mandate_issuer: MandateIssuer | None = None
    # Turns an escalation into a queued review (with the Gatekeeper only).
    referral: ReviewReferral | None = None
    # Queues a reply the hallucination check cut (with the check only).
    reply_referral: ReplyReferral | None = None


@dataclass(frozen=True)
class GuardParts:
    """Who is asking, what they may ask, and what checks the reply."""

    security: SecuritySuite
    # Loaded from persistence: the refund limit and the actor live in the database.
    agent_policy: AgentPolicy
    intent_router: CustomerSupportIntentRouter
    authenticator: CustomerAuthenticator | None = None
    orders: OrdersRepository | None = None
    deterministic_checks: DeterministicChecker | None = None
    grounding: DetectionLadder | None = None
    # Where a quarantined message waits for a reviewer.
    dead_letter_queue: DeadLetterQueue | None = None


@dataclass(frozen=True)
class MemoryParts:
    """What the agent remembers, and the window it assembles for the model."""

    working_memory: WorkingMemoryStore | None = None
    long_term_memory: MemoryFacade | None = None
    dream_recorder: TurnRecorder | None = None
    context: FiveStagePipeline | None = None
    # Absent when `context.failure_detection` is withheld.
    context_health: ContextFailureDetector | None = None
    context_strategies: StrategySelector | None = None


@dataclass(frozen=True)
class RunParts:
    """Durable execution and what each run reports."""

    # Absent only for an agent built by hand without one: then no checkpoints, no pause.
    checkpointer: SqliteCheckpointer | None = None
    runs: RunRegistry | None = None
    reporter: StageReporter | None = None
    slos: SLORegistry | None = None
    trace_exporter: TraceExporter | None = None


@dataclass(frozen=True)
class CsaParts:
    """Everything the refund agent is built from."""

    model: ModelParts
    tools: ToolParts
    guards: GuardParts
    memory: MemoryParts = field(default_factory=MemoryParts)
    run: RunParts = field(default_factory=RunParts)
    # Which stages this agent has; the declared posture when absent.
    features: FeatureSet | None = None


__all__ = ["CsaParts", "GuardParts", "MemoryParts", "ModelParts", "RunParts", "ToolParts"]
