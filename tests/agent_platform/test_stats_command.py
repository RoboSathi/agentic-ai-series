"""`/stats`: this console session's numbers, read live from the agent's own facts."""

import pytest

from agent_platform.audit import AuditEntry, DuckDbAuditLedger
from agent_platform.config import get_settings
from agent_platform.interface.commands.scope import ConsoleScope
from agent_platform.interface.commands.stats import StatsCommand
from agent_platform.logging.trace_context import bound_log_context
from agent_platform.observability.facts import (
    CLOSED,
    OPENED,
    DuckDbFactStore,
    ModelCallFact,
    TaskEventFact,
    ToolCallFact,
    TurnFact,
)
from agent_platform.observability.store import DuckDbObservabilityStore


@pytest.fixture
def store() -> DuckDbObservabilityStore:
    """Three turns: two in session "mine" (one resolved, one not), one elsewhere."""
    store = DuckDbObservabilityStore(":memory:")
    agent = get_settings().agent.id
    facts = DuckDbFactStore(store, agent_id=agent)
    for index, (session, resolved) in enumerate((("mine", True), ("mine", False), ("other", True))):
        task = f"task-{index}"
        with bound_log_context(agent=agent, session=session, trace_id=f"t-{session}"):
            facts.append(TaskEventFact(task_id=task, event=OPENED, kind="refund"))
            facts.append(TurnFact(task_id=task, status="completed", steps=1, latency_ms=50))
            facts.append(ModelCallFact(task_id=task, role="planner", model="m1", total_tokens=10))
            facts.append(
                ToolCallFact(task_id=task, tool="x.read", stage="executed", succeeded=resolved)
            )
            facts.append(
                TaskEventFact(
                    task_id=task,
                    event=CLOSED,
                    kind="refund",
                    outcome="refunded" if resolved else "abandoned",
                    success=resolved,
                    terminal=True,
                )
            )
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


def _text(store: DuckDbObservabilityStore, session: str) -> str:
    command = StatsCommand(store)
    command.bind_scope(ConsoleScope(lambda: session))
    return "\n".join(_run(command))


def _run(command: StatsCommand) -> list[str]:
    import asyncio

    return asyncio.run(command.run([])).lines


def test_only_this_sessions_work_is_counted(store: DuckDbObservabilityStore) -> None:
    text = _text(store, "mine")
    assert "Tasks · 2 · resolved 1 · open 0" in text
    assert "Turns · 2 · crashed 0 · cancelled 0 · 2 steps" in text
    assert "Model calls · 2 · 20 tokens · backups 0 · failed 0" in text
    assert "planner · m1 · 2 calls · 20 tokens" in text
    assert "Tool proposals · 2 · executed 2 · failed 1" in text
    assert "allow/VALID 1" in text


def test_slos_are_judged_against_their_targets(store: DuckDbObservabilityStore) -> None:
    words = get_settings().commands.stats
    assert f"tool_error_rate · 0.500 (target 0.050) · {words.missed}" in _text(store, "mine")


def test_a_session_with_no_turns_says_so(store: DuckDbObservabilityStore) -> None:
    command = StatsCommand(store)
    command.bind_scope(ConsoleScope(lambda: "nobody"))
    assert _run(command) == [get_settings().commands.stats.no_session]
