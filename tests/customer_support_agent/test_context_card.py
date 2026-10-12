"""The context window keeps its work and only draws a card when something needs attention.

Assembly (retrieval, budget fit, health check) is real work and stays. A healthy window is
logged and counted quietly; a window with a failure that is not routine retrieval noise
draws one warning. The logic is the platform's `ContextWindow`, shared by both agents; the
support agent's node adds only its own task metadata and grounding sources.
"""

from types import SimpleNamespace
from typing import Any

from agent_platform.config import get_settings
from agent_platform.context import ContextWindow, Task
from agent_platform.context.types import AssembledContext, Chunk, ContextFinding
from agent_platform.llm.types import CompletionRequest, Message
from agent_platform.observability import InMemoryLiveStatusFeed, StageReporter
from agent_platform.observability.llm_fields import llm_request_fields
from agent_platform.observability.types import StatusEvent
from customer_support_agent.graph.nodes.context import ContextNodes
from customer_support_agent.graph.state import AgentState

WINDOW = AssembledContext(
    chunks=[Chunk(content="Order 10975: T-shirt", kind="retrieved", source="10975")],
    budget_tokens=3000,
    used_tokens=486,
)
POISONED = ContextFinding(
    mode="poisoning", source="10975", rule="injected instruction", detail="x", action="dropped"
)
ROUTINE = ContextFinding(
    mode="distraction", source="kb", rule="off task", detail="x", action="dropped"
)


def _pipeline() -> Any:
    async def assemble(
        _task: Any, budget_tokens: int | None = None, strategies: Any = None
    ) -> AssembledContext:
        return WINDOW

    return SimpleNamespace(assemble=assemble)


def _health(findings: list[ContextFinding]) -> Any:
    async def diagnose(window: Any, task: Any) -> list[ContextFinding]:
        return findings

    return SimpleNamespace(
        diagnose=diagnose,
        mitigate=lambda window, found: (window, found),
        rot=SimpleNamespace(window_used=lambda session, tokens: None),
    )


def _reporter() -> tuple[StageReporter, list[StatusEvent]]:
    feed = InMemoryLiveStatusFeed(max_events=0)
    shown: list[StatusEvent] = []
    feed.subscribe(shown.append)
    return StageReporter(feed, "t"), shown


async def _build(findings: list[ContextFinding]) -> list[StatusEvent]:
    reporter, shown = _reporter()
    window = ContextWindow(_pipeline(), health=_health(findings))
    await window.build(Task(query="refund"), reporter)
    return [e for e in shown if e.status != "started"]


class TestAHealthyWindowIsQuiet:
    async def test_no_visible_card(self) -> None:
        assert await _build([]) == []

    async def test_routine_noise_already_removed_draws_no_warning(self) -> None:
        assert await _build([ROUTINE]) == []


class TestAPoisonedWindowIsNot:
    async def test_a_finding_that_is_not_routine_draws_one_warning(self) -> None:
        [warning] = await _build([POISONED])
        assert warning.status == "warning" and "poisoning" in (warning.summary or "")
        words = get_settings().context.window
        assert "486 of 3000 tokens" in warning.fields[words.size_title][0]
        lines = warning.fields[words.findings_title]
        assert all(line["tone"] == "fail" for line in lines)
        assert any("poisoning" in line["text"] for line in lines)
        assert any("REMOVED: all of 10975" in line["text"] for line in lines)


class _Nodes(ContextNodes):
    """`ContextNodes` with its collaborators stubbed and a recording reporter."""

    def __init__(self) -> None:
        self._context: Any = _pipeline()
        self._strategies: Any = None
        self._context_health = None
        self._reporter, _ = _reporter()


class TestTheSupportAgentsNode:
    async def test_the_decision_line_survives(self) -> None:
        state = AgentState(query="refund", session_id="s", customer_id=89)
        result = await _Nodes().assemble_context(state)
        assert "context: 486/3000 tokens" in result["decisions"]


class TestTheLlmRequestSaysHowBigItIs:
    def test_the_context_size_is_in_tokens(self) -> None:
        request = CompletionRequest(
            messages=[
                Message(role="system", content="x" * 4000),
                Message(role="user", content="refund"),
            ]
        )
        assert llm_request_fields(request, model="m")["context_tokens"] >= 1000
