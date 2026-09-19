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
from agentic_commons.config import get_settings
from agentic_commons.context import (
    Chunk,
    CrossEncoderReranker,
    DenseRetriever,
    FiveStagePipeline,
    HybridRetriever,
    LexicalRetriever,
    RelevanceRanker,
    Task,
)
from agentic_commons.evals import NollmScoringCascadeImpl, create_scoring_cascade
from agentic_commons.gatekeeper import GatekeeperImpl
from agentic_commons.gatekeeper.ledger import CumulativeLedger
from agentic_commons.hallucination_protection import DetectionLadder
from agentic_commons.llm.client import LLMClient, create_llm_client
from agentic_commons.llm.encoders import create_reranker
from agentic_commons.memory import MemoryFacade
from agentic_commons.observability import SLORegistry, StageReporter
from agentic_commons.observability.tracer import LocalTracer
from agentic_commons.persistence import SqliteConnectionPool
from agentic_commons.resilience import SqliteIdempotencyGuard
from agentic_commons.security import Authorizer, PIIFilterImpl, SecuritySuite
from agentic_commons.tools import NamespacedToolRegistry, ToolRetrievalSelector
from customer_support_agent.core.authentication import CustomerAuthenticator
from customer_support_agent.core.graph import CustomerSupportAgent
from customer_support_agent.core.identity import current_customer_id
from customer_support_agent.core.refund_saga import RefundSaga
from customer_support_agent.core.repositories import (
    HitlQueueRepository,
    OrdersRepository,
    RefundLedgerRepository,
    UsersRepository,
)
from customer_support_agent.core.review import ReviewQueueService
from customer_support_agent.core.schema import Order, policy_text
from customer_support_agent.core.tools import SupportTools


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


class WorkingMemoryContextSource:
    """Contributes what this conversation has already established.

    Pinned, and for the same reason the policy is: these are not
    retrieval results competing on relevance, they are the facts of the
    conversation in progress. An agent that drops "this customer is
    verified as #41" under budget pressure asks for the code again.

    This is the source that makes working memory *used* rather than
    merely written. Without it the window is a store nothing reads, which
    is how it sat for several episodes.
    """

    def __init__(self, memory: MemoryFacade) -> None:
        self._memory = memory

    async def candidates(self, task: Task) -> list[Chunk]:
        """This session's active facts, newest first."""
        return [
            Chunk(
                content=f"{fact.key}: {fact.value}",
                kind="memory",
                source=fact.provenance.source,
                pinned=True,
            )
            for fact in self._memory.active_window(task.session_id)
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


class MemoryContextSource:
    """Contributes recalled memory to the window."""

    def __init__(self, memory: MemoryFacade) -> None:
        self._memory = memory

    async def candidates(self, task: Task) -> list[Chunk]:
        """Facts recalled for this query."""
        from agentic_commons.memory import MemoryQuery

        recalled = await self._memory.recall(
            MemoryQuery(text=task.query, session_id=task.session_id)
        )
        return [
            Chunk(
                content=f"{scored.fact.key}: {scored.fact.value}",
                kind="memory",
                source=scored.fact.provenance.source,
                score=scored.score,
            )
            for scored in recalled
        ]


def _order_chunk(order: Order) -> Chunk:
    """One order, as a sentence a retriever can match against."""
    return Chunk(
        content=(
            f"Order {order.order_id} placed {order.timestamp[:10]}: {order.order_status}, "
            f"${order.amount:,.2f}, refund {order.refund_status}"
        ),
        kind="retrieved",
        source=str(order.order_id),
    )


def build_agent(
    llm: LLMClient | None = None,
    pool: SqliteConnectionPool | None = None,
    gatekeeper: GatekeeperImpl | None = None,
    memory: MemoryFacade | None = None,
    cascade: NollmScoringCascadeImpl | None = None,
    audit: AuditLedger | None = None,
    reporter: StageReporter | None = None,
    with_memory: bool | None = None,
    observe: bool | None = None,
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
    connection = pool or SqliteConnectionPool()
    scoring = cascade or create_scoring_cascade()
    remember = settings.memory.enabled if with_memory is None else with_memory
    observing = settings.observability.enabled if observe is None else observe

    orders = OrdersRepository(connection)
    hitl = HitlQueueRepository(connection)
    refunds = RefundLedgerRepository(connection)

    saga = RefundSaga(orders, refunds, idempotency=SqliteIdempotencyGuard(connection))
    tools = SupportTools(orders, hitl, refunds, saga)

    registry = NamespacedToolRegistry()
    registry.register_all(tools.definitions())

    facade = (
        memory if memory is not None else (MemoryFacade(relevance=scoring) if remember else None)
    )

    sources: list[object] = [PolicyContextSource(), OrderCorpusContextSource(orders, scoring)]
    if facade is not None:
        sources.append(WorkingMemoryContextSource(facade))
        sources.append(MemoryContextSource(facade))
    # Two-pass relevance: the NOLLM cascade scores every candidate with an
    # embedding model, then a cross-encoder rescores the shortlist. The
    # reranker is None when no hosted encoder is reachable, and the
    # pipeline then keeps the cheaper ordering rather than pretending.
    reranker = create_reranker(fallback=scoring)
    # With context assembly off, the agent still gets its policy — the
    # graph falls back to it. An agent reasoning without its own refund
    # policy is not a narrower agent, it is a wrong one.
    context = (
        FiveStagePipeline(
            sources=sources,  # type: ignore[arg-type]
            ranker=scoring,
            reranker=CrossEncoderReranker(reranker) if reranker is not None else None,
        )
        if settings.context.enabled
        else None
    )

    return CustomerSupportAgent(
        llm=llm or create_llm_client(),
        gatekeeper=gatekeeper
        or GatekeeperImpl(
            authorizer=Authorizer(),
            ledger=CumulativeLedger(connection),
            audit=audit or create_audit_ledger(connection),
        ),
        registry=registry,
        tools=tools,
        security=SecuritySuite(),
        memory=facade,
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
        tracer=LocalTracer(connection, redactor=PIIFilterImpl()) if observing else None,
        slos=SLORegistry() if observing else None,
        # Grounding answers "is this reply supported by what the tools
        # returned?", which is a hallucination-protection question and
        # not an observability one. Gating it on `observe` meant an agent
        # configured to be quiet was also an agent that stopped checking
        # its own answers.
        grounding=DetectionLadder(scoring) if settings.hallucination_protection.enabled else None,
        authenticator=CustomerAuthenticator(UsersRepository(connection), orders),
        orders=orders,
    )


def build_review_service(
    pool: SqliteConnectionPool | None = None, audit: AuditLedger | None = None
) -> ReviewQueueService:
    """The operator-side entry point for the human review queue.

    Built separately from the agent on purpose: this is the human half
    of the escalation path, and the agent must have no route to it.
    """
    connection = pool or SqliteConnectionPool()
    orders = OrdersRepository(connection)
    refunds = RefundLedgerRepository(connection)
    return ReviewQueueService(
        queue=HitlQueueRepository(connection),
        orders=orders,
        saga=RefundSaga(orders, refunds, idempotency=SqliteIdempotencyGuard(connection)),
        audit=audit or create_audit_ledger(connection),
    )


def build_lexical_index(pool: SqliteConnectionPool) -> LexicalRetriever:
    """A BM25 index over the order corpus, for identifier-shaped lookups."""
    orders = OrdersRepository(pool)
    return LexicalRetriever(
        [
            Chunk(
                content=(
                    f"Order {order.order_id} for {order.user_id}: {order.order_status}, "
                    f"${order.amount:.2f}, refund {order.refund_status}"
                ),
                kind="retrieved",
                source=str(order.order_id),
            )
            for order in orders.query()
        ]
    )


def database_path() -> str:
    """Where the agent's database lives, per config."""
    return str(get_settings().persistence.resolved_db_path)
