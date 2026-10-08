"""`/stats`: this console session's numbers, read live from the agent's own ledgers."""

import pytest

from agent_platform.audit import AuditEntry, DuckDbAuditLedger
from agent_platform.commands.scope import ConsoleScope
from agent_platform.commands.stats import StatsCommand
from agent_platform.config import get_settings
from agent_platform.logging.trace_context import bound_log_context
from agent_platform.observability.facts import (
    DuckDbFactLedger,
    LlmCallFact,
    ToolCallFact,
    TurnOutcomeFact,
)
from agent_platform.observability.store import DuckDbObservabilityStore


@pytest.fixture
def store() -> DuckDbObservabilityStore:
    store = DuckDbObservabilityStore(":memory:")
    agent = get_settings().agent.id
    facts = DuckDbFactLedger(store, agent_id=agent)
    for session, answered in (("mine", True), ("mine", False), ("other", True)):
        with bound_log_context(agent=agent, session=session, trace_id=f"t-{session}"):
            facts.record(LlmCallFact(role="planner", model="m1", total_tokens=10, latency_ms=50))
            facts.record(ToolCallFact(action="x.read", succeeded=answered))
            facts.record(TurnOutcomeFact(outcome="o", answered=answered, escalated=not answered))
    DuckDbAuditLedger(store, agent_id=agent).record(
        AuditEntry(
            trace_id="t-mine",
            actor=agent,
            action="x.read",
            authority_context={"session_id": "mine", "standing": "VALID"},
            decision="allow",
        )
    )
    return store


def _command(store: DuckDbObservabilityStore, session: str) -> StatsCommand:
    command = StatsCommand(store)
    command.bind_scope(ConsoleScope(lambda: session))
    return command


async def test_only_this_sessions_turns_are_counted(store: DuckDbObservabilityStore) -> None:
    text = "\n".join((await _command(store, "mine").run([])).lines)
    assert "Turns · 2 · answered 1 · escalated 1" in text
    assert "Model calls · 2 · 20 tokens" in text
    assert "planner · m1 · 2 calls" in text
    assert "Tool calls · 2 · failed 1" in text
    assert "allow/VALID 1" in text


async def test_slos_are_judged_against_their_targets(store: DuckDbObservabilityStore) -> None:
    text = "\n".join((await _command(store, "mine").run([])).lines)
    words = get_settings().commands.stats
    assert f"task_success_rate · 0.500 (target 0.950) · {words.missed}" in text


async def test_a_session_with_no_turns_says_so(store: DuckDbObservabilityStore) -> None:
    result = await _command(store, "nobody").run([])
    assert result.lines == [get_settings().commands.stats.no_session]
