"""The support agent's profile and procedures, end to end (ADR 0010).

Built by the real composition root against the seeded database with its deployed policy;
only the model is scripted, so these assert what the model was actually sent, how many
times it was asked, and what the trail and the feed say about it.
"""

import pytest

from agent_platform.audit import AuditEntry, DuckDbAuditLedger
from agent_platform.config import FeatureSet, get_settings
from agent_platform.observability import InMemoryLiveStatusFeed, StageReporter
from agent_platform.observability.store import DuckDbObservabilityStore
from agent_platform.observability.types import StatusEvent
from agent_platform.persistence import SqliteConnectionPool
from customer_support_agent.evals.fixture import policy_seed_sql
from customer_support_agent.graph import build_agent
from tests.conftest import ScriptedLLM, tool_call

_REFUND = "I want a refund for order 10001"
# Withheld so the turn is hermetic: the other memory classes reach an embedding model, and
# the screens and the grounding check reach hosted models — none of them is under test here.
_WITHHELD = (
    "memory.semantic",
    "memory.episodic",
    "content_moderation",
    "prompt_injection",
    "pii",
    "hallucination_guard",
)


def _features(procedural: bool) -> FeatureSet:
    features = get_settings().features.with_override("memory.procedural", procedural)
    for path in _WITHHELD:
        features = features.with_override(path, False)
    return features


class _Turn:
    def __init__(self, llm: ScriptedLLM, events: list[StatusEvent]) -> None:
        self.llm = llm
        self.events = events
        self.route = ""

    def card(self) -> StatusEvent:
        [card] = self.stage(get_settings().skills.observability.activation_stage)
        return card

    def system(self) -> str:
        return str(self.llm.requests[0].messages[0].content)

    def offered(self) -> list[str]:
        return [tool["function"]["name"] for tool in self.llm.requests[0].tools or []]

    def stage(self, stage: str) -> list[StatusEvent]:
        return [e for e in self.events if e.stage == stage and e.status != "started"]


@pytest.fixture(autouse=True)
def deployed_policy(seeded_pool: SqliteConnectionPool) -> None:
    """The policy the agent deploys with (role, permissions), from the `make dummy-data` seed."""
    seeded_pool.execute_script(policy_seed_sql())


@pytest.fixture
def audit() -> DuckDbAuditLedger:
    return DuckDbAuditLedger(DuckDbObservabilityStore(":memory:"), agent_id=get_settings().agent.id)


async def _refund_turn(
    pool: SqliteConnectionPool,
    audit: DuckDbAuditLedger,
    *,
    procedural: bool,
    query: str = _REFUND,
) -> _Turn:
    llm = ScriptedLLM(
        [
            {"tool_calls": tool_call("process_refund", order_id=10001, amount=32.0)},
            {"content": "Your refund has been issued and settles shortly."},
        ]
    )
    feed = InMemoryLiveStatusFeed(max_events=0)
    events: list[StatusEvent] = []
    feed.subscribe(events.append)
    agent = build_agent(
        llm=llm,
        pool=pool,
        audit=audit,
        features=_features(procedural),
        reporter=StageReporter(feed, run_id="csa-procedures"),
    )
    state = await agent.handle(query)
    assert llm.requests, "the scripted model was never called"
    turn = _Turn(llm, events)
    turn.route = str(state.metadata.get("intent_execution_route", ""))
    return turn


def _activations(audit: DuckDbAuditLedger) -> list[AuditEntry]:
    action = get_settings().skills.audit.activation_action
    return [entry for entry in audit.recent(100) if entry.action == action]


class TestARefundTurn:
    async def test_the_procedure_saves_the_model_its_reading_rounds(
        self, seeded_pool: SqliteConnectionPool, audit: DuckDbAuditLedger
    ) -> None:
        """Two model calls — request the refund, then reply — not four."""
        turn = await _refund_turn(seeded_pool, audit, procedural=True)
        assert len(turn.llm.requests) == 2
        first = turn.llm.requests[0].messages
        asked = [c.name for m in first if m.tool_calls for c in m.tool_calls]
        assert asked == ["get_order_status", "get_refund_status"]
        assert all("not permitted" not in str(m.content) for m in first if m.role == "tool")

    async def test_it_is_offered_only_what_the_turn_still_needs(
        self, seeded_pool: SqliteConnectionPool, audit: DuckDbAuditLedger
    ) -> None:
        """The reads are already answered and the order is known: only the write is left."""
        turn = await _refund_turn(seeded_pool, audit, procedural=True)
        assert turn.offered() == ["process_refund"]

    async def test_the_order_the_checks_read_is_not_read_again(
        self, seeded_pool: SqliteConnectionPool, audit: DuckDbAuditLedger
    ) -> None:
        """Prefetch answers both reads from the deterministic checks' row; no tool runs for them."""
        turn = await _refund_turn(seeded_pool, audit, procedural=True)
        served = [e for e in turn.stage("tool_result") if e.fields.get("served_from")]
        requested = [e.fields["tool"] for e in turn.stage("tool_request")]
        assert [e.fields["tool"] for e in served] == ["get_order_status", "get_refund_status"]
        assert requested == ["process_refund"]
        assert "10001" in served[0].fields["output"]

    async def test_it_carries_the_profile_and_the_refund_procedure(
        self, seeded_pool: SqliteConnectionPool, audit: DuckDbAuditLedger
    ) -> None:
        system = (await _refund_turn(seeded_pool, audit, procedural=True)).system()
        assert system.startswith("# Customer Support Agent")
        assert "# Procedure: refund-workflow" in system
        assert "full total, whatever it is" in system
        assert "## Preconditions" not in system  # the people's sections are not sent

    async def test_one_procedure_card_says_what_ran_and_why(
        self, seeded_pool: SqliteConnectionPool, audit: DuckDbAuditLedger
    ) -> None:
        turn = await _refund_turn(seeded_pool, audit, procedural=True)
        stage = get_settings().skills.observability.activation_stage
        [card] = turn.stage(stage)
        assert card.summary is not None
        assert card.summary.startswith("refund-workflow v1 · chosen for intent REQUEST_REFUND")
        assert "harness reads first: get_order_status, get_refund_status" in card.summary
        assert card.summary.endswith("model may call: process_refund")
        prefetched = [
            e
            for e in [*turn.stage("tool_request"), *turn.stage("tool_result")]
            if e.fields.get("prefetched_for")
        ]
        assert sorted(e.fields["tool"] for e in prefetched) == [
            "get_order_status",
            "get_refund_status",
        ]

    async def test_the_audit_trail_names_the_version_used(
        self, seeded_pool: SqliteConnectionPool, audit: DuckDbAuditLedger
    ) -> None:
        await _refund_turn(seeded_pool, audit, procedural=True)
        [entry] = _activations(audit)
        assert entry.authority_context["procedures"] == ["refund-workflow@1"]
        assert entry.authority_context["profile"].startswith("profile@")


class TestTheCardTellsTheTruth:
    async def test_it_comes_before_any_read_it_explains(
        self, seeded_pool: SqliteConnectionPool, audit: DuckDbAuditLedger
    ) -> None:
        """No tool call appears without the procedure that asked for it, drawn first."""
        turn = await _refund_turn(seeded_pool, audit, procedural=True)
        stage = get_settings().skills.observability.activation_stage
        order = [e.stage for e in turn.events if e.status != "started"]
        assert order.index(stage) < order.index("tool_request")
        assert turn.card().fields["harness_reads_first"] == [
            "get_order_status",
            "get_refund_status",
        ]

    async def test_it_lists_exactly_the_tools_the_model_was_offered(
        self, seeded_pool: SqliteConnectionPool, audit: DuckDbAuditLedger
    ) -> None:
        turn = await _refund_turn(seeded_pool, audit, procedural=True)
        assert sorted(turn.card().fields["model_may_call"]) == sorted(turn.offered())


class TestAnUnplacedTurn:
    """The classifier could not place it and it names no order: reads only."""

    _UNPLACED = "something is not right, can you look into it"

    async def test_it_gets_the_order_status_procedure_only(
        self, seeded_pool: SqliteConnectionPool, audit: DuckDbAuditLedger
    ) -> None:
        turn = await _refund_turn(seeded_pool, audit, procedural=True, query=self._UNPLACED)
        assert turn.route == "llm_fallback"
        names = [p["procedure"].split(" v")[0] for p in turn.card().fields["procedures"]]
        assert names == ["order-status"]

    async def test_it_is_offered_reads_and_the_card_says_so(
        self, seeded_pool: SqliteConnectionPool, audit: DuckDbAuditLedger
    ) -> None:
        """A write needs a request the router recognised as one."""
        turn = await _refund_turn(seeded_pool, audit, procedural=True, query=self._UNPLACED)
        assert sorted(turn.offered()) == ["get_order_status", "list_orders"]
        assert sorted(turn.card().fields["model_may_call"]) == sorted(turn.offered())


class TestWithoutProceduralMemory:
    async def test_the_profile_is_all_there_is_and_nothing_is_prefetched(
        self, seeded_pool: SqliteConnectionPool, audit: DuckDbAuditLedger
    ) -> None:
        turn = await _refund_turn(seeded_pool, audit, procedural=False)
        assert turn.system().startswith("# Customer Support Agent")
        assert "# Procedure:" not in turn.system()
        assert _activations(audit) == []
        assert not [e for e in turn.stage("tool_request") if e.fields.get("prefetched_for")]


def test_no_intent_prompt_outgrows_the_old_instructions() -> None:
    """Profile + the largest procedures that fit one turn stay under the old prompt's 3,167."""
    from agent_platform.skills import load_profile, load_skill_documents

    settings = get_settings().skills
    sizes = sorted((len(d.prompt) for d in load_skill_documents(settings)), reverse=True)
    worst = len(load_profile(settings).text) + sum(sizes[: settings.selection.max_active])
    assert 0 < settings.max_prompt_chars < 3167
    assert worst <= settings.max_prompt_chars
