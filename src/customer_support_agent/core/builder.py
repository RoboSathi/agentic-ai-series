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
    FiveStagePipeline,
    LexicalRetriever,
    Task,
)
from agentic_commons.evals import NollmScoringCascadeImpl, create_scoring_cascade
from agentic_commons.gatekeeper import GatekeeperImpl
from agentic_commons.gatekeeper.ledger import CumulativeLedger
from agentic_commons.llm.client import LLMClient, create_llm_client
from agentic_commons.memory import MemoryFacade
from agentic_commons.observability import StageReporter
from agentic_commons.persistence import SqliteConnectionPool
from agentic_commons.resilience import SqliteIdempotencyGuard
from agentic_commons.security import Authorizer, SecuritySuite
from agentic_commons.tools import NamespacedToolRegistry, ToolRetrievalSelector
from customer_support_agent.core.graph import CustomerSupportAgent
from customer_support_agent.core.refund_saga import RefundSaga
from customer_support_agent.core.repositories import (
    HitlQueueRepository,
    OrdersRepository,
    RefundLedgerRepository,
)
from customer_support_agent.core.schema import policy_text
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


def build_agent(
    llm: LLMClient | None = None,
    pool: SqliteConnectionPool | None = None,
    gatekeeper: GatekeeperImpl | None = None,
    memory: MemoryFacade | None = None,
    cascade: NollmScoringCascadeImpl | None = None,
    audit: AuditLedger | None = None,
    reporter: StageReporter | None = None,
    with_memory: bool = True,
) -> CustomerSupportAgent:
    """Wire the refund agent from config, honouring any injected parts."""
    connection = pool or SqliteConnectionPool()
    scoring = cascade or create_scoring_cascade()

    orders = OrdersRepository(connection)
    hitl = HitlQueueRepository(connection)
    refunds = RefundLedgerRepository(connection)

    saga = RefundSaga(orders, refunds, idempotency=SqliteIdempotencyGuard(connection))
    tools = SupportTools(orders, hitl, refunds, saga)

    registry = NamespacedToolRegistry()
    registry.register_all(tools.definitions())

    facade = (
        memory if memory is not None else (MemoryFacade(relevance=scoring) if with_memory else None)
    )

    sources: list[object] = [PolicyContextSource()]
    if facade is not None:
        sources.append(MemoryContextSource(facade))
    context = FiveStagePipeline(sources=sources, ranker=scoring)  # type: ignore[arg-type]

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
        selector=ToolRetrievalSelector(registry, scoring),
        cascade=scoring,
        reporter=reporter,
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
                source=order.order_id,
            )
            for order in orders.query()
        ]
    )


def database_path() -> str:
    """Where the agent's database lives, per config."""
    return str(get_settings().persistence.resolved_db_path)
