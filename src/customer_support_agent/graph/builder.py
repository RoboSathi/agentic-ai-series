"""The composition root (LLD §21).

One place where every concrete class is chosen. Everywhere else takes
what it needs as a constructor argument, which is what makes the graph
testable against a scripted LLM and an in-memory database without a
single patched import.

Anything passed in is used as-is; anything omitted is built from config.
So a test overrides the two things it cares about and gets the real
implementations of everything else — the wiring under test stays the
wiring that ships.
"""

from agent_platform.audit import AuditLedger, create_audit_ledger
from agent_platform.config import (
    FeatureSet,
    active_features,
    get_settings,
    set_active_features,
)
from agent_platform.context import (
    Chunk,
    CrossEncoderReranker,
    DenseRetriever,
    FiveStagePipeline,
    HybridRetriever,
    LexicalRetriever,
    RelevanceRanker,
    Task,
    WorkingMemoryContextSource,
)
from agent_platform.context.factory import build_context_management
from agent_platform.context.types import TRUST
from agent_platform.deterministic_checks import DeterministicChecker
from agent_platform.durability import (
    RunRegistry,
    SqliteCheckpointer,
    create_checkpointer,
    model_types,
)
from agent_platform.evals import NollmScoringCascadeImpl, create_scoring_cascade
from agent_platform.executor import ToolExecutorImpl
from agent_platform.gatekeeper import Gatekeeper, GatekeeperImpl
from agent_platform.gatekeeper.ledger import CumulativeLedger
from agent_platform.hallucination_check import DetectionLadder
from agent_platform.host.memory import long_term_memory
from agent_platform.llm.client import LLMClient, create_llm_client
from agent_platform.llm.local_reranker import create_reranker
from agent_platform.loop_control import LoopBudget, build_loop_governor
from agent_platform.memory import (
    EpisodicStore,
    FileDreamInbox,
    TurnRecorder,
    WorkingMemoryStore,
)
from agent_platform.memory.context_source import MemoryRecallContextSource
from agent_platform.memory.procedural.skills import build_agent_instructions
from agent_platform.memory.procedural.store import SkillStore
from agent_platform.memory.semantic.rag import (
    KnowledgeBase,
    KnowledgeBaseContextSource,
    create_knowledge_base,
)
from agent_platform.observability import (
    DuckDbObservabilityStore,
    SLORegistry,
    StageReporter,
)
from agent_platform.observability.facts import create_fact_store, install_fact_store
from agent_platform.observability.langsmith import build_trace_exporter
from agent_platform.observability.model_observer import StageReporterModelObserver
from agent_platform.observability.tracer import LocalTracer
from agent_platform.observability.units import OutcomeCatalog
from agent_platform.persistence import SqliteConnectionPool, SqlitePlatformStore
from agent_platform.resilience import CompensatingSaga, DeadLetterQueue, SqliteIdempotencyGuard
from agent_platform.security import (
    Authorizer,
    PIIFilterImpl,
    PromptInjectionGuard,
    SecuritySuite,
    TermContentModerator,
)
from agent_platform.security.auth.policy import AgentPolicy, AgentPolicyRepository
from agent_platform.security.pii.boundary import bounded
from agent_platform.security.prompt_injection.detector import InstructionDataSeparator
from agent_platform.tools import NamespacedToolRegistry, ToolRetrievalSelector
from agent_platform.verification import ReadBackVerifier
from customer_support_agent.config import get_agent_settings
from customer_support_agent.deterministic_checks.order_validator import OrderChecks
from customer_support_agent.domain.identity import current_customer_id
from customer_support_agent.domain.models import Order
from customer_support_agent.domain.order_reference import (
    ORDER_UNDER_DISCUSSION,
    WRITE_INTENT,
)
from customer_support_agent.domain.refund_policy import RefundPolicy
from customer_support_agent.gatekeeper.business_rules import (
    REFUND_ACTION_CLASS,
    CustomerSupportBusinessRules,
)
from customer_support_agent.gatekeeper.mandates import EstablishedOrderScope
from customer_support_agent.graph.agent import CustomerSupportAgent
from customer_support_agent.graph.state import AgentState
from customer_support_agent.intent.router import CustomerSupportIntentRouter
from customer_support_agent.persistence.repositories import (
    HitlQueueRepository,
    OrdersRepository,
    RefundLedgerRepository,
    UsersRepository,
)
from customer_support_agent.persistence.schema import ensure_schema, require_seeded
from customer_support_agent.presentation.orders import order_sentence
from customer_support_agent.resilience.cancellation_saga import CancellationSaga
from customer_support_agent.resilience.refund_saga import RefundSaga
from customer_support_agent.resilience.refund_settlement import RefundSettlement
from customer_support_agent.review.queue import ReviewQueueService
from customer_support_agent.review.referral import ReplyReferral, ReviewReferral
from customer_support_agent.security.authentication import CustomerAuthenticator
from customer_support_agent.tools.definitions import CustomerSupportToolHandler, SupportTools
from customer_support_agent.verification import RefundEffectSource

# Used only when this agent declares no budgeted action at all, in which
# case the window never matters because nothing is charged against it.
DEFAULT_WINDOW_HOURS = 24


class OrderCorpusContextSource:
    """Contributes the customer's own orders, retrieved (E19, FR-7).

    Hybrid retrieval — BM25 for the identifier-shaped queries a support
    conversation is full of ("10001"), embeddings for the ones that
    describe rather than name ("the parcel that never turned up"). Both
    are needed: lexical search cannot match a paraphrase and dense search
    is unreliable on an exact id, which is the pairing E19 exists to
    show.

    **Scoped to the authenticated customer, every time.** The corpus is
    rebuilt per request from that customer's orders rather than filtered
    afterwards: a retriever over every customer's orders is one ranking
    bug away from putting somebody else's purchase in this customer's
    context window, and no amount of post-filtering makes that safe if
    the filter is ever forgotten.

    **Retrieval is the fallback, not the first answer.** When the turn
    has already settled which order is under discussion, there is nothing
    to rank: that order is the context, and it is returned alone. Ranking
    anyway is how "I want a refund" — a sentence naming no order, so
    nothing for BM25 to match — produced five of the customer's *other*
    orders and omitted the one they meant. Eleven one-line orders that
    differ only in a product name and a price are near-indistinguishable
    to a dense retriever; asking it to choose between them on a query
    that describes none of them is asking for an arbitrary answer.
    """

    def __init__(self, orders: OrdersRepository, ranker: RelevanceRanker) -> None:
        self._orders = orders
        self._ranker = ranker
        self._top_k = get_settings().context.top_k

    async def candidates(self, task: Task) -> list[Chunk]:
        """The order under discussion, or the ones most relevant to this query."""
        customer_id = current_customer_id()
        if customer_id is None:
            # No identity, no corpus. An unauthenticated request retrieves
            # nothing rather than everything.
            return []
        corpus = [_order_chunk(order) for order in self._orders.for_user(customer_id)]
        if not corpus:
            return []
        established = self._established(task, corpus)
        if established is not None:
            return [established]
        if task.metadata.get(WRITE_INTENT):
            # A refund or cancellation is about one order. With none settled,
            # ranking would put other orders' states in front of the model;
            # it resolves the order with `list_orders` instead.
            return []
        retriever = HybridRetriever(
            lexical=LexicalRetriever(corpus),
            dense=DenseRetriever(self._ranker, corpus),
        )
        return await retriever.retrieve(task.query, self._top_k)

    def _established(self, task: Task, corpus: list[Chunk]) -> Chunk | None:
        """The chunk for the order this turn settled on, if it settled on one.

        Matched within the corpus that was already scoped to this
        customer, never fetched by id: an order this customer does not
        own cannot become their context by being named.
        """
        order_id = task.metadata.get(ORDER_UNDER_DISCUSSION)
        if isinstance(order_id, bool) or not isinstance(order_id, int):
            return None
        return next((chunk for chunk in corpus if chunk.source == str(order_id)), None)


def _order_chunk(order: Order) -> Chunk:
    """One order, as a sentence a retriever can match against.

    The product name is in here because a customer describes rather than
    names: "the parcel that never turned up" and "my headphones" are the
    queries dense retrieval exists for, and neither matches a row that
    says only an id, a status and an amount.
    """
    # Read from the system of record this turn: verified, not a retrieved copy.
    return Chunk(
        content=order_sentence(order),
        kind="retrieved",
        source=str(order.order_id),
        metadata={TRUST: "verified"},
    )


def _cumulative_ledger(
    platform_store: SqlitePlatformStore,
    agent_id: str,
    policy: AgentPolicy,
    ledger_enabled: bool,
) -> CumulativeLedger:
    """The rolling budget this agent's write actions are charged against.

    Both halves come from outside this function. Which tool belongs to
    which budget class is the agent's mapping, read from its config; what
    each class may spend is the agent's persisted policy. A literal
    `{"process_refund": "refund"}` here was a policy decision hiding in
    wiring — the next budgeted tool would have had to be added in Python
    rather than in configuration.
    """
    limits = policy.limits.values()
    return CumulativeLedger(
        platform_store,
        agent_id=agent_id,
        enabled=ledger_enabled,
        action_classes=get_agent_settings().action_classes,
        thresholds={limit.action_class: limit.window for limit in limits},
        window_hours=min((limit.window_hours for limit in limits), default=DEFAULT_WINDOW_HOURS),
    )


def build_agent(
    llm: LLMClient | None = None,
    pool: SqliteConnectionPool | None = None,
    gatekeeper: Gatekeeper | None = None,
    working_memory: WorkingMemoryStore | None = None,
    cascade: NollmScoringCascadeImpl | None = None,
    audit: AuditLedger | None = None,
    reporter: StageReporter | None = None,
    with_memory: bool | None = None,
    observe: bool | None = None,
    deterministic_checks: DeterministicChecker | None = None,
    intent_router: CustomerSupportIntentRouter | None = None,
    features: FeatureSet | None = None,
    knowledge_base: KnowledgeBase | None = None,
    checkpointer: SqliteCheckpointer | None = None,
    episodic_store: EpisodicStore | None = None,
) -> CustomerSupportAgent:
    """Wire the refund agent from config, honouring any injected parts.

    Which subsystems are wired is a configuration decision, not a literal
    in this function. Each optional collaborator below is gated on its own
    `enabled` flag, so a narrower config produces a genuinely narrower
    agent — the same graph with fewer things in it, which is what lets one
    episode run the real product with only that episode's subsystem live.

    Leaving a collaborator out is deliberately not the same as letting it
    switch itself off internally. A disabled component still constructed
    is still called, still traced, and still narrates a step to anyone
    watching; one that was never built has no step to narrate. The
    difference is the whole point on screen.

    The `with_memory` and `observe` arguments override their config flags
    for a caller that knows what it wants — a test, mostly. `None` means
    "ask the configuration", which is the normal case.
    """
    settings = get_settings()
    agent_settings = get_agent_settings()
    # The one place a feature flag is read. Everything below is wiring,
    # not policy — which is what lets two agents with different postures
    # run in one process: pass a set explicitly and this deployment is
    # composed from it, whatever any other agent was given.
    # What a console last set, else what this agent declared. Reading only
    # the declared config here meant `/harness off` rebuilt the agent with
    # every feature back on: the command moved one store and the builder
    # read another.
    features = features if features is not None else active_features(settings.features)
    set_active_features(features)
    agent_id = settings.agent.id
    connection = pool or SqliteConnectionPool(agent_settings.resolved_db_path)
    # Idempotent; upgrades a database built before reply reviews shared the queue.
    ensure_schema(connection)
    if pool is None:
        require_seeded(connection, agent_settings.resolved_db_path)
    platform_store = SqlitePlatformStore(connection, agent_id=agent_id)
    policy_repo = AgentPolicyRepository(platform_store)
    agent_policy = policy_repo.get()
    scoring = cascade or create_scoring_cascade()
    memory_enabled = features.is_on("memory") if with_memory is None else with_memory
    observing = settings.observability.enabled if observe is None else observe

    # One shared analytics connection per running agent process. The store is
    # intentionally separate from the agent's relational/business SQLite
    # database: every agent writes its observability here and identifies its
    # records with agent_id.
    audit_needs_store = audit is None and settings.audit.enabled
    observability_store = DuckDbObservabilityStore() if (observing or audit_needs_store) else None
    # Tasks, turns, model calls, tool proposals and checks, as rows the analytics
    # source aggregates: written by the platform's choke points, never by this agent.
    install_fact_store(create_fact_store(observability_store if observing else None, agent_id))
    # Every task kind and outcome this agent names is declared, or it does not start.
    tasks = get_agent_settings().tasks
    OutcomeCatalog().require(kinds=list(tasks.kinds), outcomes=tasks.outcomes())

    orders = OrdersRepository(connection)
    hitl = HitlQueueRepository(connection)
    refunds = RefundLedgerRepository(connection)
    # The review rule needs the agent's direct limit: a refund is queued for
    # a person only when it is above what the agent may refund itself.
    refund_policy = RefundPolicy(
        orders, direct_limit=agent_policy.per_action_limit(REFUND_ACTION_CLASS)
    )

    idempotency = SqliteIdempotencyGuard(platform_store)
    dead_letter_queue = DeadLetterQueue(
        platform_store,
        agent_id=agent_id,
        enabled=settings.resilience.dead_letter_queue.enabled,
    )
    saga = RefundSaga(orders, refunds, idempotency=idempotency, saga=CompensatingSaga())
    cancellation = CancellationSaga(orders, saga, idempotency=idempotency, saga=CompensatingSaga())
    tools = SupportTools(
        orders,
        hitl,
        refunds,
        saga,
        refund_policy=refund_policy,
        cancellation=cancellation,
        list_limit=agent_settings.list_orders_max_rows,
    )

    registry = NamespacedToolRegistry()
    registry.register_all(tools.definitions())
    authorizer = Authorizer(policy=agent_policy)

    # DeterministicCheck is the one feature whose *rules* are the agent's own.
    # `agent_platform` owns only the contract — `DeterministicChecker`, a
    # Protocol taking a request and returning findings — and never learns
    # what an order or a refund is. Customer Support supplies
    # `OrderChecks`; another agent built on this platform declares
    # `deterministic_checks` in its own config and injects its own checker here,
    # with nothing in the platform to change.
    tool_handler = CustomerSupportToolHandler(tools)
    deterministic_checker = (
        deterministic_checks or OrderChecks(orders, refund_policy=refund_policy)
        if features.is_on("deterministic_checks")
        else None
    )
    # Intent routing is an agent-owned component, but the concrete provider is
    # selected by the shared classifier factory. The graph receives the router
    # by dependency injection and never constructs one itself.
    resolved_intent_router = intent_router or CustomerSupportIntentRouter()

    # PII redaction reaches tool-call payloads, traces, and memory events.
    redactor = PIIFilterImpl() if features.is_on("pii") else None
    live_working_memory = (
        working_memory
        if working_memory is not None
        else (
            WorkingMemoryStore(max_history=agent_settings.history_turns) if memory_enabled else None
        )
    )
    # Every completed turn leaves one dream event, redacted, for `make dream csa` (ADR 0012).
    dream_recorder = (
        TurnRecorder(FileDreamInbox(), agent_id, redact=redactor.redact if redactor else None)
        if live_working_memory is not None
        else None
    )
    ledger = audit or create_audit_ledger(store=observability_store, agent_id=agent_id)
    # Long-term memory: facts about the customer (semantic) and what
    # happened (episodic), each built only if declared. Every operation is
    # governed — provenance derived from who is speaking, RBAC against the
    # agent's persisted policy plus the platform's role grants, audited,
    # and shown as a card. Redaction happens at the write, when PII is on.
    # Procedural memory: the agent's skills and the lessons a curator
    # approved. One store, shared by the facade (approvals land in it) and
    # the instructions (each turn's skills are read from it).
    procedural = SkillStore() if memory_enabled and features.is_on("memory.procedural") else None
    long_term = (
        long_term_memory(
            features,
            platform_store=platform_store,
            procedural=procedural,
            redactor=redactor,
            policy=agent_policy,
            audit=ledger,
            relevance=scoring,
            reporter=reporter,
            episodic_store=episodic_store,
        )
        if memory_enabled
        else None
    )

    # No policy source. The policy reaches the model in the system prompt,
    # which is always present and never trimmed; contributing a second
    # verbatim copy here spent a third of the window restating it.
    # Two-pass relevance: the NOLLM cascade scores every candidate with an
    # embedding model, then a cross-encoder rescores the shortlist. The
    # reranker is None when no hosted encoder is reachable, and the
    # pipeline then keeps the cheaper ordering rather than pretending.
    # The second pass is its own capability: with `context.rerank` off the
    # pipeline keeps the cheap ordering, which is the E19 contrast.
    reranker = create_reranker(fallback=scoring) if features.is_on("context.rerank") else None
    sources: list[object] = [OrderCorpusContextSource(orders, scoring)]
    if live_working_memory is not None:
        sources.append(WorkingMemoryContextSource(live_working_memory))
    # Semantic memory is this agent's knowledge corpus, retrieved into the
    # window by hybrid search. Withheld, the source is never built and
    # the agent answers policy questions from its system prompt alone.
    if memory_enabled and features.is_on("memory.semantic"):
        sources.append(
            KnowledgeBaseContextSource(knowledge_base or create_knowledge_base(reranker=reranker))
        )
    # What this customer has told the agent before, recalled cheapest tier
    # first. Scoped to the verified customer by the task, never wider.
    if long_term is not None:
        sources.append(MemoryRecallContextSource(long_term))
    # With context assembly off, the agent still gets its policy — the
    # graph falls back to it. An agent reasoning without its own refund
    # policy is not a narrower agent, it is a wrong one.
    # The strategies, the health check (poisoning uses the security layer's own phrase list,
    # so the two never disagree about what an injection is) and the offload fetch tool.
    management = build_context_management(
        features, live_working_memory, screen=InstructionDataSeparator()
    )
    if management.fetch_tool is not None:
        registry.register(management.fetch_tool)
    context = (
        FiveStagePipeline(
            sources=sources,  # type: ignore[arg-type]
            ranker=scoring,
            reranker=CrossEncoderReranker(reranker) if reranker is not None else None,
        )
        if features.is_on("context")
        else None
    )

    # Withholding the Gatekeeper is the sharpest demonstration in the set:
    # the same refund that was escalated a moment ago simply happens. Only
    # the authority decision goes: the call still runs through the Executor
    # (idempotency, deadline, audit), and its audit row reads UNGOVERNED.
    governed = (
        gatekeeper
        or GatekeeperImpl(
            authorizer=authorizer,
            policy=agent_policy,
            ledger=_cumulative_ledger(
                platform_store, agent_id, agent_policy, settings.gatekeeper.ledger.enabled
            ),
            audit=ledger,
            business_rules=CustomerSupportBusinessRules(agent_policy),
        )
        if features.is_on("gatekeeper")
        else None
    )

    return CustomerSupportAgent(
        # Every provider call tokenises personal data out and restores it in the answer.
        llm=bounded(
            llm or create_llm_client(observer=StageReporterModelObserver(lambda: reporter)),
            features,
            redactor,
        ),
        gatekeeper=governed,
        # Only a Gatekeeper escalates, so only a governed agent refers.
        referral=ReviewReferral() if governed is not None else None,
        # Only the hallucination check cuts a reply, so only a checked agent refers one.
        reply_referral=ReplyReferral(hitl) if features.is_on("hallucination_guard") else None,
        audit=ledger,
        features=features,
        # Exported traces are always PII-redacted, whatever the `pii` feature says.
        trace_exporter=build_trace_exporter(agent_id, lambda: PIIFilterImpl().redact),
        registry=registry,
        tools=tools,
        executor=ToolExecutorImpl(
            handler=tool_handler,
            idempotency=idempotency,
            reporter=reporter,
        ),
        dead_letter_queue=dead_letter_queue,
        # Each screening detector is wired only if this agent declared
        # it. Absent means the stage does not run and has nothing to
        # narrate — not a detector that runs and reports finding nothing.
        security=SecuritySuite(
            authorizer=authorizer,
            pii=redactor,
            injection=PromptInjectionGuard() if features.is_on("prompt_injection") else None,
            moderator=(TermContentModerator() if features.is_on("content_moderation") else None),
        ),
        working_memory=live_working_memory,
        dream_recorder=dream_recorder,
        long_term_memory=long_term,
        # Measured on what was actually assembled, and mitigated before the
        # model sees it. Poisoning uses the security layer's own phrase
        # list, so the two can never disagree about what an injection is.
        context_health=management.health,
        context_strategies=management.strategies,
        # The loop's caps are this agent's own; the baseline is its own turn history.
        governor=build_loop_governor(
            LoopBudget(max_steps=agent_settings.max_tool_iterations),
            facts=observability_store if observing else None,
            agent_id=agent_id,
        ),
        context=context,
        instructions=build_agent_instructions(
            settings.skills, store=procedural, audit=ledger, reporter=reporter, tools=registry
        ),
        # Retrieval-based tool selection narrows within what the intent
        # already permits. Without it the agent is offered every tool the
        # intent allows, which is the pre-E22 behaviour and exactly what
        # that episode exists to contrast against.
        selector=ToolRetrievalSelector(registry, scoring) if settings.tools.enabled else None,
        cascade=scoring,
        reporter=reporter,
        # Observability is on by default: a production agent that has to
        # be reconfigured before it can be diagnosed is diagnosed late.
        # The redactor is not optional. A span carries the query, the
        # tool arguments and the result, so an unredacted trace is a
        # copy of the customer's data in a second place — and FR-44
        # says there is no path that writes one. There was: this one.
        tracer=(
            LocalTracer(
                redactor=PIIFilterImpl(),
                store=observability_store,
                agent_id=agent_id,
            )
            if observing
            else None
        ),
        slos=SLORegistry() if observing else None,
        # Grounding answers "is this reply supported by what the tools
        # returned?", which is a hallucination-protection question and
        # not an observability one. Gating it on `observe` meant an agent
        # configured to be quiet was also an agent that stopped checking
        # its own answers.
        grounding=DetectionLadder(scoring) if features.is_on("hallucination_guard") else None,
        # Every side effect is read back before anything is claimed about
        # it. Not a feature toggle: an agent that cannot tell whether its
        # own refund landed is not a narrower agent, it is an unreliable
        # one, and "the executor returned success" was never evidence the
        # money moved.
        verifier=ReadBackVerifier(RefundEffectSource(orders, refunds, hitl)),
        # Only built when the Gatekeeper is: a scope with nothing to
        # check it is a record nobody reads.
        mandate_issuer=EstablishedOrderScope(orders) if governed is not None else None,
        authenticator=CustomerAuthenticator(UsersRepository(connection)),
        orders=orders,
        deterministic_checks=deterministic_checker,
        agent_policy=agent_policy,
        intent_router=resolved_intent_router,
        # Durable execution is not a feature: without it a crash loses the
        # turn and a refund waiting on a person loses the conversation that
        # asked for it. That is a broken program, not a narrower agent.
        checkpointer=checkpointer or create_checkpointer(state_types=model_types(AgentState)),
        runs=RunRegistry(platform_store),
    )


def build_review_service(
    pool: SqliteConnectionPool | None = None, audit: AuditLedger | None = None
) -> ReviewQueueService:
    """The operator-side entry point for the human review queue.

    Built separately from the agent on purpose: this is the human half
    of the escalation path, and the agent must have no route to it.
    """
    settings = get_settings()
    agent_settings = get_agent_settings()
    connection = pool or SqliteConnectionPool(agent_settings.resolved_db_path)
    platform_store = SqlitePlatformStore(connection, agent_id=settings.agent.id)
    orders = OrdersRepository(connection)
    refunds = RefundLedgerRepository(connection)
    observability_store = DuckDbObservabilityStore() if get_settings().audit.enabled else None
    idempotency = SqliteIdempotencyGuard(platform_store)
    saga = RefundSaga(orders, refunds, idempotency=idempotency, saga=CompensatingSaga())
    return ReviewQueueService(
        queue=HitlQueueRepository(connection),
        orders=orders,
        saga=saga,
        cancellation=CancellationSaga(
            orders, saga, idempotency=idempotency, saga=CompensatingSaga()
        ),
        audit=audit
        or create_audit_ledger(
            store=observability_store,
            agent_id=get_settings().agent.id,
        ),
    )


def build_settlement_service(
    pool: SqliteConnectionPool | None = None, audit: AuditLedger | None = None
) -> RefundSettlement:
    """Voiding and settling issued refunds (`/void`, `/settle`, the console's sweep).

    The operator's side of a VOIDABLE refund, built apart from the agent
    like the review queue: the agent issues refunds and has no route to
    void or settle one. Its idempotency guard is the agent's own, so a void
    can release the records the refund was made under.
    """
    settings = get_settings()
    connection = pool or SqliteConnectionPool(get_agent_settings().resolved_db_path)
    platform_store = SqlitePlatformStore(connection, agent_id=settings.agent.id)
    observability_store = DuckDbObservabilityStore() if settings.audit.enabled else None
    return RefundSettlement(
        OrdersRepository(connection),
        RefundLedgerRepository(connection),
        idempotency=SqliteIdempotencyGuard(platform_store),
        audit=audit or create_audit_ledger(store=observability_store, agent_id=settings.agent.id),
    )


def build_lexical_index(pool: SqliteConnectionPool) -> LexicalRetriever:
    """A BM25 index over the order corpus, for identifier-shaped lookups."""
    orders = OrdersRepository(pool)
    return LexicalRetriever([_order_chunk(order) for order in orders.query()])


def database_path() -> str:
    """Where the agent's database lives, per config."""
    return str(get_agent_settings().resolved_db_path)
