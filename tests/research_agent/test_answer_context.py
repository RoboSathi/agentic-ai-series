"""The direct answer's context: working memory, through the platform's context window when
`context` is declared, and as plain blocks when it is withheld. Nothing from an earlier run."""

from types import SimpleNamespace
from typing import Any

import pytest

from agent_platform.context import (
    ContextWindow,
    FiveStagePipeline,
    StrategySelector,
    WorkingMemoryContextSource,
)
from agent_platform.memory.working import WorkingMemoryStore
from agent_platform.observability import InMemoryLiveStatusFeed, StageReporter
from research_agent.config import get_agent_settings
from research_agent.context.session import AnswerContext
from tests.research_agent.conftest import BuildHarness

pytestmark = pytest.mark.usefixtures("ra_config")


def _working() -> WorkingMemoryStore:
    settings = get_agent_settings().answer
    working = WorkingMemoryStore(settings.context.history_turns)
    working.get("s").add_history("weather in Pune?", "Light drizzle, 26 °C.")
    working.get("s").set("run_status", "COMPLETE")
    return working


def _context(window: bool) -> tuple[AnswerContext, list[Any]]:
    settings = get_agent_settings().answer
    working = _working()
    memory: Any = SimpleNamespace(working=working)
    feed = InMemoryLiveStatusFeed(max_events=0)
    shown: list[Any] = []
    feed.subscribe(shown.append)
    built = (
        ContextWindow(
            FiveStagePipeline(sources=[WorkingMemoryContextSource(working)]),
            StrategySelector(),
        )
        if window
        else None
    )
    reporter = StageReporter(feed, "t")
    context = AnswerContext(
        memory,
        settings.context,
        settings.messages,
        window=built,
        reporter=lambda: reporter,
    )
    return context, shown


async def test_the_window_carries_the_turns_and_the_state() -> None:
    context, shown = _context(window=True)
    session = await context.assemble("s", "what limits iron-air battery cycle life?")
    assert session.window is not None and session.blocks == []
    text = session.window.text
    assert "Light drizzle" in text and "run_status" in text
    assert session.history == []  # the turns are inside the window, not separate messages
    assert any(e.stage == "context_assembly" for e in shown)


async def test_withheld_the_turns_go_as_the_conversation_they_were() -> None:
    context, shown = _context(window=False)
    session = await context.assemble("s", "what limits iron-air battery cycle life?")
    assert [m.role for m in session.history] == ["user", "assistant"]
    assert any("run_status" in block for block in session.blocks)
    assert not shown


async def test_the_interview_is_remembered_in_working_memory(harness: BuildHarness) -> None:
    gate = {"decision": "scopable", "mode": "research", "archetype": "general", "topic": "t"}
    question = {"next_question": "Who is this for?", "next_item_id": "audience"}
    h = harness({"interviewer": [gate, question]})
    await h.agent.handle("/research the cycle life of iron-air cells", "s")
    memory = h.agent._memory  # the agent's own working memory
    assert memory is not None and memory.working is not None
    history = memory.working.get("s").history
    assert history[-2]["content"] == "the cycle life of iron-air cells"
    assert "Who is this for?" in history[-1]["content"]
