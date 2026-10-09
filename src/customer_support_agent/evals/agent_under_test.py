"""The Customer Support agent, driven one golden case at a time.

Every case gets a private world: its own copy of the fixture database
(orders, refunds, review queue, platform policy, semantic memory), its
own audit ledger and checkpoint file, and its own episodic log. Nothing
one case does — a refund paid, a fact remembered — is visible to the
next. What is shared is only what is expensive and stateless: the LLM
client and the intent router.

The agent is the real one (`build_agent`), with the same features, model
and guards the product runs with. The only difference is where it keeps
its state.
"""

import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

from agent_platform.audit import SqliteAuditLedger
from agent_platform.config import get_settings
from agent_platform.durability import create_checkpointer, model_types
from agent_platform.evals.contracts import GoldenCase, ToolCallRecord, Transcript, UsageRecord
from agent_platform.llm.client import LLMClient, create_llm_client
from agent_platform.llm.recording import RecordingLLMClient
from agent_platform.logging.trace_context import bound_channel
from agent_platform.memory import EpisodicStore
from agent_platform.observability import InMemoryLiveStatusFeed, StageReporter
from agent_platform.observability.types import StatusEvent
from agent_platform.persistence import SqliteConnectionPool
from customer_support_agent.config.settings import CsaEvalSettings, get_agent_settings
from customer_support_agent.domain.identity import Customer, set_current_customer
from customer_support_agent.evals.dataset.schema import CsaCasePayload
from customer_support_agent.evals.fixture import EvalFixture, FixtureFacts
from customer_support_agent.graph.state import AgentState
from customer_support_agent.intent.router import CustomerSupportIntentRouter

_TOOL_REQUEST = "tool_request"
_TOOL_RESULT = "tool_result"
_GATEKEEPER = "gatekeeper"
_ARGUMENT_PREFIX = "arg_"
_NOT_EXECUTED = "not_executed"
# A reason code is an UPPER_SNAKE token under a key that names a code.
_REASON_CODE = re.compile(r"^[A-Z][A-Z0-9_]{3,}$")
_CODE_KEY = "code"
_EVIDENCE_KEY = "order_id"
# A refusal is as much a looked-up record as a success.
_EVIDENCE_STATUSES = frozenset({"completed", "blocked"})
_METADATA_KEYS = (
    "halted",
    "intent",
    "intent_disposition",
    "awaiting_review",
    "refusal_is_final",
    "hallucination_rate",
    "guardrail_reason_code",
    "guardrail_escalated",
)


class CsaAgentUnderTest:
    """Runs one golden case against a freshly isolated agent."""

    def __init__(
        self,
        fixture: EvalFixture,
        run_key: str,
        llm: LLMClient | None = None,
        classifier: Any = None,
        settings: CsaEvalSettings | None = None,
    ) -> None:
        self._fixture = fixture
        self._facts: FixtureFacts = fixture.facts()
        self._run_key = run_key
        self._llm = llm
        self._classifier = classifier
        self._settings = settings or get_agent_settings().evals
        self._router: CustomerSupportIntentRouter | None = None

    async def run(self, case: GoldenCase) -> Transcript:
        payload = CsaCasePayload.of(case)
        database = self._fixture.copy_for(self._run_key, case.case_id)
        workspace = database.with_suffix("")
        try:
            return await self._run_isolated(case, payload, database, workspace)
        finally:
            self._fixture.discard(database)
            if not self._settings.keep_case_workspaces:
                shutil.rmtree(workspace, ignore_errors=True)

    async def _run_isolated(
        self, case: GoldenCase, payload: CsaCasePayload, database: Path, workspace: Path
    ) -> Transcript:
        pool = SqliteConnectionPool(database)
        checkpointer = create_checkpointer(
            state_types=model_types(AgentState), path=workspace / "checkpoints.db"
        )
        feed = InMemoryLiveStatusFeed()
        llm = RecordingLLMClient(self._shared_llm())
        agent = self._build(pool, checkpointer, feed, llm, case, workspace)
        customer = self._customer(payload.customer)
        started = time.perf_counter()
        replies: list[str] = []
        states: list[AgentState] = []
        try:
            for turn in case.turns:
                # Eval traffic is told apart by its channel (ADR 0009 D5).
                with bound_channel(get_settings().evals.channel):
                    state = await agent.handle(
                        turn, session_id=self._session(case), customer=customer
                    )
                states.append(state)
                replies.append(state.response)
            return self._transcript(case, replies, states, feed.events, llm, pool, started)
        finally:
            set_current_customer(None)
            checkpointer.conn.close()
            pool.close()

    def _build(
        self,
        pool: SqliteConnectionPool,
        checkpointer: Any,
        feed: InMemoryLiveStatusFeed,
        llm: LLMClient,
        case: GoldenCase,
        workspace: Path,
    ) -> Any:
        from customer_support_agent.graph import build_agent

        return build_agent(
            llm=llm,
            pool=pool,
            audit=SqliteAuditLedger(pool),
            reporter=StageReporter(feed, self._session(case)),
            # Tracing and SLOs go to the shared analytics file the operator's
            # console holds open; they change nothing the agent decides.
            observe=False,
            intent_router=self._intent_router(),
            checkpointer=checkpointer,
            episodic_store=EpisodicStore(root=workspace / "episodic"),
        )

    def _transcript(
        self,
        case: GoldenCase,
        replies: list[str],
        states: list[AgentState],
        events: list[StatusEvent],
        llm: RecordingLLMClient,
        pool: SqliteConnectionPool,
        started: float,
    ) -> Transcript:
        roles = frozenset(self._settings.grounding_roles)
        return Transcript(
            suite=case.suite,
            case_id=case.case_id,
            replies=replies,
            tool_calls=tool_calls_from(events),
            evidence=[*llm.grounding_messages(roles), *record_evidence(events)],
            side_effects=side_effects(pool, self._facts),
            metadata=_metadata(states, events),
            usage=UsageRecord(
                llm_calls=llm.calls,
                prompt_tokens=llm.prompt_tokens,
                completion_tokens=llm.completion_tokens,
                cost_usd=llm.cost_usd,
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
            ),
        )

    def _customer(self, code: str) -> Customer:
        user = self._facts.user(code)
        return Customer(customer_id=user.user_id, name=user.name, email=user.email)

    def _session(self, case: GoldenCase) -> str:
        return f"{self._settings.session_prefix}-{case.case_id}"

    def _shared_llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = create_llm_client()
        return self._llm

    def _intent_router(self) -> CustomerSupportIntentRouter:
        if self._router is None:
            self._router = (
                CustomerSupportIntentRouter(backend=self._classifier)
                if self._classifier is not None
                else CustomerSupportIntentRouter()
            )
        return self._router


def tool_calls_from(events: list[StatusEvent]) -> list[ToolCallRecord]:
    """Rebuild each tool call — request, gatekeeper ruling, result — from the feed."""
    calls: list[ToolCallRecord] = []
    for event in events:
        fields = event.fields
        if event.event_kind == _TOOL_REQUEST:
            calls.append(
                ToolCallRecord(
                    name=str(fields.get("qualified_tool", "")),
                    arguments={
                        key.removeprefix(_ARGUMENT_PREFIX): value
                        for key, value in fields.items()
                        if key.startswith(_ARGUMENT_PREFIX)
                    },
                    status=_NOT_EXECUTED,
                )
            )
        elif calls and event.event_kind == _GATEKEEPER and "decision" in fields:
            calls[-1].decision = str(fields["decision"])
            calls[-1].reason_code = str(fields.get("reason_code", ""))
        elif calls and event.event_kind == _TOOL_RESULT:
            calls[-1].status = str(fields.get("execution_status") or fields.get("result", ""))
            calls[-1].output = json.dumps(fields, default=str)
    return calls


def record_evidence(events: list[StatusEvent]) -> list[str]:
    """Every record the agent looked up, including deterministic (no-LLM) answers.

    A record can sit anywhere in an event — a sanity finding nests the
    order it checked inside its evidence — so the whole field tree is searched.
    """
    return [
        json.dumps(event.fields, default=str, sort_keys=True)
        for event in events
        if event.status in _EVIDENCE_STATUSES and _mentions(event.fields, _EVIDENCE_KEY)
    ]


def _mentions(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_mentions(child, key) for child in value.values())
    if isinstance(value, list):
        return any(_mentions(child, key) for child in value)
    return False


def side_effects(pool: SqliteConnectionPool, facts: FixtureFacts) -> dict[str, Any]:
    """What this case changed in the books, relative to the fixture."""
    refunds = [
        dict(row)
        for row in pool.execute(
            "SELECT refund_id, order_id, user_id, amount_cents, reversed FROM refunds"
        ).rows
        if row["refund_id"] not in facts.seeded_refund_ids
    ]
    reviews = [
        dict(row)
        for row in pool.execute(
            # Money reviews only: a reply review is not a side effect the oracle scores.
            "SELECT id, order_id, user_id, requested_amount_cents, status FROM hitl_queue "
            "WHERE action != 'reply'"
        ).rows
        if row["id"] not in facts.seeded_review_ids
    ]
    changed = [
        {
            "order_id": row["order_id"],
            "before": facts.orders[row["order_id"]].refund_status,
            "after": row["refund_status"],
        }
        for row in pool.execute("SELECT order_id, refund_status FROM orders").rows
        if row["order_id"] in facts.orders
        and facts.orders[row["order_id"]].refund_status != row["refund_status"]
    ]
    return {"refunds": refunds, "review_requests": reviews, "order_changes": changed}


def reason_codes(events: list[StatusEvent], states: list[AgentState]) -> list[str]:
    """Every policy / guard reason code the run surfaced, in order, once each."""
    found: dict[str, None] = {}
    for event in events:
        _collect_codes(event.fields, found)
    for state in states:
        _collect_codes(state.metadata, found)
    return list(found)


def _collect_codes(value: Any, found: dict[str, None], key: str = "") -> None:
    if isinstance(value, dict):
        for child_key, child in value.items():
            _collect_codes(child, found, str(child_key))
    elif isinstance(value, list):
        for child in value:
            _collect_codes(child, found, key)
    elif isinstance(value, str) and _CODE_KEY in key.casefold() and _REASON_CODE.match(value):
        found.setdefault(value, None)


def _metadata(states: list[AgentState], events: list[StatusEvent]) -> dict[str, Any]:
    last = states[-1] if states else AgentState()
    return {
        **{key: _signal(key, last, events) for key in _METADATA_KEYS},
        "escalated": any(state.escalated for state in states),
        "decisions": [decision for state in states for decision in state.decisions],
        "tool_rounds": sum(state.tool_rounds for state in states),
        "reason_codes": reason_codes(events, states),
    }


def _signal(key: str, state: AgentState, events: list[StatusEvent]) -> Any:
    """A signal from the final state, else the latest event that reported it.

    The grounding verdict is reported on the feed as it is decided; the
    final state a durable run hands back does not always carry it.
    """
    if state.metadata.get(key) is not None:
        return state.metadata[key]
    for event in reversed(events):
        if event.fields.get(key) is not None:
            return event.fields[key]
    return None
