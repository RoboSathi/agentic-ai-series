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

from agentic_commons.audit import AuditLedger, create_audit_ledger
from agentic_commons.basic_sanity import BasicSanityChecker
from agentic_commons.config import (
    FeatureSet,
    active_features,
    get_settings,
    set_active_features,
)
from agentic_commons.context import (
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
from agentic_commons.evals import NollmScoringCascadeImpl, create_scoring_cascade
from agentic_commons.executor import ToolExecutorImpl
from agentic_commons.gatekeeper import Gatekeeper, GatekeeperImpl
from agentic_commons.gatekeeper.ledger import CumulativeLedger
from agentic_commons.hallucination_protection import DetectionLadder
from agentic_commons.llm.client import LLMClient, create_llm_client
from agentic_commons.llm.encoders import create_reranker
from agentic_commons.memory import DreamInbox, WorkingMemoryStore
from agentic_commons.observability import (
    DuckDbObservabilityStore,
    SLORegistry,
    StageReporter,
)
from agentic_commons.observability.tracer import LocalTracer
from agentic_commons.persistence import SqliteConnectionPool, SqlitePlatformStore
from agentic_commons.resilience import DeadLetterQueue, SqliteIdempotencyGuard
from agentic_commons.security import (
    Authorizer,
    PIIFilterImpl,
    PromptInjectionGuard,
    SecuritySuite,
    TermContentModerator,
)
from agentic_commons.security.auth.policy import AgentPolicy, AgentPolicyRepository
from agentic_commons.tools import NamespacedToolRegistry, ToolRetrievalSelector
from agentic_commons.verification import ReadBackVerifier
from customer_support_agent.basic_sanity.order_validator import OrderSanityCheck
from customer_support_agent.config import get_agent_settings
from customer_support_agent.domain.identity import current_customer_id
from customer_support_agent.domain.models import Order
from customer_support_agent.domain.refund_policy import RefundPolicy
from customer_support_agent.gatekeeper.business_rules import CustomerSupportBusinessRules
from customer_support_agent.gatekeeper.mandates import RefundMandateIssuer
from customer_support_agent.graph.agent import CustomerSupportAgent
from customer_support_agent.intent.router import CustomerSupportIntentRouter
from customer_support_agent.persistence.repositories import (
    HitlQueueRepository,
    OrdersRepository,
    RefundLedgerRepository,
    UsersRepository,
)
from customer_support_agent.prompts.policy import policy_text
from customer_support_agent.resilience.refund_saga import RefundSaga
from customer_support_agent.review.queue import ReviewQueueService
from customer_support_agent.security.authentication import CustomerAuthenticator
from customer_support_agent.tools.definitions import CustomerSupportToolHandler, SupportTools
from customer_support_agent.verification import RefundEffectSource

# Used only when this agent declares no budgeted action at all, in which
# case the window never matters because nothing is charged against it.
DEFAULT_WINDOW_HOURS = 24


class PolicyContextSource:
    """Contributes the refund policy to every assembled window.

    Pinned, because the policy is not a retrieval result competing on
    relevance — an agent that drops its own policy under budget pressure
    is an agent operating without one.
    """

    def __init__(self) -> None:
        self._policy = policy_text()

    async def candidates(self, task: Task) -> list[Chunk]:
        """The policy, as one pinned chunk."""
        return [
            Chunk(content=self._policy, kind="policy", source="customer_policy.md", pinned=True)
        ]


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
    """

    def __init__(self, orders: OrdersRepository, ranker: RelevanceRanker) -> None:
        self._orders = orders
        self._ranker = ranker
        self._top_k = get_settings().context.top_k

    async def candidates(self, task: Task) -> list[Chunk]:
        """The customer's orders most relevant to this query."""
        customer_id = current_customer_id()
        if customer_id is None:
            # No identity, no corpus. An unauthenticated request retrieves
            # nothing rather than everything.
            return []
        corpus = [_order_chunk(order) for order in self._orders.for_user(customer_id)]
        if not corpus:
            return []
        retriever = HybridRetriever(
            lexical=LexicalRetriever(corpus),
            dense=DenseRetriever(self._ranker, corpus),
        )
        return await retriever.retrieve(task.query, self._top_k)


def _order_chunk(order: Order) -> Chunk:
    """One order, as a sentence a retriever can match against.

    The product name is in here because a customer describes rather than
    names: "the parcel that never turned up" and "my headphones" are the
    queries dense retrieval exists for, and neither matches a row that
    says only an id, a status and an amount.
    """
    return Chunk(
        content=(
            f"Order {order.order_id} placed {order.timestamp[:10]}: {order.description}, "
            f"{order.order_status}, {order.whereabouts}, ${order.amount:,.2f}, "
            f"refund {order.refund_status}"
        ),
        kind="retrieved",
        source=str(order.order_id),
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
    basic_sanity: BasicSanityChecker | None = None,
    intent_router: CustomerSupportIntentRouter | None = None,
    features: FeatureSet | None = None,
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
    observability_store = (
        DuckDbObservabilityStore()
        if (observing or audit_needs_store)
        else None
    )

    orders = OrdersRepository(connection)
    hitl = HitlQueueRepository(connection)
    refunds = RefundLedgerRepository(connection)
    refund_policy = RefundPolicy(orders)

    idempotency = SqliteIdempotencyGuard(platform_store)
    dead_letter_queue = DeadLetterQueue(
        platform_store,
        agent_id=agent_id,
        enabled=settings.resilience.dead_letter_queue.enabled,
    )
    saga = RefundSaga(
        orders,
        refunds,
        idempotency=idempotency,
        refund_policy=refund_policy,
    )
    tools = SupportTools(orders, hitl, refunds, saga, refund_policy=refund_policy)

    registry = NamespacedToolRegistry()
    registry.register_all(tools.definitions())
    authorizer = Authorizer(policy=agent_policy)

    # BasicSanity is the one feature whose *rules* are the agent's own.
    # `agentic_commons` owns only the contract — `BasicSanityChecker`, a
    # Protocol taking a request and returning findings — and never learns
    # what an order or a refund is. Customer Support supplies
    # `OrderSanityCheck`; another agent built on this platform declares
    # `basic_sanity` in its own config and injects its own checker here,
    # with nothing in the platform to change.
    tool_handler = CustomerSupportToolHandler(tools)
    sanity_checker = (
        basic_sanity or OrderSanityCheck(orders, refund_policy=refund_policy)
        if features.is_on("basic_sanity")
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
        else (WorkingMemoryStore(max_history=agent_settings.history_turns) if memory_enabled else None)
    )
    memory_event_sink = DreamInbox(platform_store) if live_working_memory is not None else None

    sources: list[object] = [PolicyContextSource(), OrderCorpusContextSource(orders, scoring)]
    if live_working_memory is not None:
        sources.append(WorkingMemoryContextSource(live_working_memory))
    # Two-pass relevance: the NOLLM cascade scores every candidate with an
    # embedding model, then a cross-encoder rescores the shortlist. The
    # reranker is None when no hosted encoder is reachable, and the
    # pipeline then keeps the cheaper ordering rather than pretending.
    # The second pass is its own capability: with `context.rerank` off the
    # pipeline keeps the cheap ordering, which is the E19 contrast.
    reranker = create_reranker(fallback=scoring) if features.is_on("context.rerank") else None
    # With context assembly off, the agent still gets its policy — the
    # graph falls back to it. An agent reasoning without its own refund
    # policy is not a narrower agent, it is a wrong one.
    context = (
        FiveStagePipeline(
            sources=sources,  # type: ignore[arg-type]
            ranker=scoring,
            reranker=CrossEncoderReranker(reranker) if reranker is not None else None,
        )
        if features.is_on("context")
        else None
    )

    ledger = audit or create_audit_ledger(store=observability_store, agent_id=agent_id)
    # Withholding the Gatekeeper is the sharpest demonstration in the set:
    # the same refund that was escalated a moment ago simply happens. The
    # audit entry is still written on that path, so the trail records an
    # ungoverned execution rather than recording nothing at all.
    governed = (
        gatekeeper
        or GatekeeperImpl(
            authorizer=authorizer,
            policy=agent_policy,
            ledger=_cumulative_ledger(
                platform_store, agent_id, agent_policy, settings.gatekeeper.ledger.enabled
            ),
            audit=ledger,
            business_rules=CustomerSupportBusinessRules(
                orders, hitl, agent_policy, refund_policy=refund_policy
            ),
        )
        if features.is_on("gatekeeper")
        else None
    )

    return CustomerSupportAgent(
        llm=llm or create_llm_client(),
        gatekeeper=governed,
        audit=ledger,
        features=features,
        registry=registry,
        tools=tools,
        executor=ToolExecutorImpl(
            handler=tool_handler,
            idempotency=idempotency,
            reporter=reporter,
        ),
        # The same adapter, unwrapped. With the Gatekeeper withheld the
        # graph calls this directly, so there is no idempotency key and no
        # deadline — which is precisely what the execution boundary was
        # providing, shown by its absence.
        handler=tool_handler,
        dead_letter_queue=dead_letter_queue,
        # Each screening detector is wired only if this agent declared
        # it. Absent means the stage does not run and has nothing to
        # narrate — not a detector that runs and reports finding nothing.
        security=SecuritySuite(
            authorizer=authorizer,
            pii=redactor,
            injection=PromptInjectionGuard() if features.is_on("prompt_injection") else None,
            moderator=(
                TermContentModerator() if features.is_on("content_moderation") else None
            ),
        ),
        working_memory=live_working_memory,
        memory_event_sink=memory_event_sink,
        context=context,
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
        # Only built when the Gatekeeper is: a mandate with nothing
        # to check it is a record nobody reads.
        mandate_issuer=RefundMandateIssuer(orders) if governed is not None else None,
        authenticator=CustomerAuthenticator(UsersRepository(connection), orders),
        orders=orders,
        basic_sanity=sanity_checker,
        agent_policy=agent_policy,
        intent_router=resolved_intent_router,
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
    observability_store = (
        DuckDbObservabilityStore() if get_settings().audit.enabled else None
    )
    return ReviewQueueService(
        queue=HitlQueueRepository(connection),
        orders=orders,
        saga=RefundSaga(orders, refunds, idempotency=SqliteIdempotencyGuard(platform_store)),
        audit=audit
        or create_audit_ledger(
            store=observability_store,
            agent_id=get_settings().agent.id,
        ),
    )


def build_lexical_index(pool: SqliteConnectionPool) -> LexicalRetriever:
    """A BM25 index over the order corpus, for identifier-shaped lookups."""
    orders = OrdersRepository(pool)
    return LexicalRetriever(
        [
            _order_chunk(order) for order in orders.query()
        ]
    )


def database_path() -> str:
    """Where the agent's database lives, per config."""
    return str(get_agent_settings().resolved_db_path)
