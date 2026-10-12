"""The support agent's Cancel → Resume, as the console's buttons drive it.

A turn cancelled while the model is answering stops at once, says where it
stopped, and is offered back as "Resume Agent from <node>"; resumed, it finishes
from that node with the customer bound again.
"""

import asyncio
from typing import Any

from agent_platform.core.persistence import SqliteConnectionPool
from agent_platform.llm.types import CompletionRequest, CompletionResult
from customer_support_agent.console import CustomerSupportHost
from customer_support_agent.domain.identity import current_customer, set_current_customer
from customer_support_agent.graph import build_agent
from tests.conftest import CUSTOMER_CODE, ScriptedLLM


class _HeldLLM(ScriptedLLM):
    """Answers, but holds its first answer until released (the moment Cancel is pressed)."""

    def __init__(self) -> None:
        super().__init__([{"content": "Order 10001 was delivered."}])
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        self.entered.set()
        await self.release.wait()
        return await super().complete(request)


def _host(agent: Any) -> CustomerSupportHost:
    host = CustomerSupportHost()
    host._agent = agent
    return host


async def _cancelled_turn(pool: SqliteConnectionPool) -> tuple[Any, _HeldLLM, Any]:
    set_current_customer(None)
    llm = _HeldLLM()
    agent = build_agent(llm=llm, pool=pool, with_memory=False)
    llm.release.set()
    await agent.handle(CUSTOMER_CODE, session_id="s")
    llm.release.clear()
    turn = asyncio.ensure_future(agent.handle("I want a refund for order 10001", session_id="s"))
    entered = asyncio.ensure_future(llm.entered.wait())
    await asyncio.wait({turn, entered}, return_when=asyncio.FIRST_COMPLETED)
    assert entered.done() and not turn.done(), "the turn must be waiting on the model"
    assert _host(agent).actions("s")[0].command == "/cancel"
    assert agent.cancel() is True
    return agent, llm, await turn


class TestCancelThenResume:
    async def test_a_cancelled_turn_says_where_it_stopped(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        agent, _, state = await _cancelled_turn(seeded_pool)
        assert "Stopped before" in state.response
        last = agent.runs.latest_in_session("s")
        assert last is not None and last.status == "cancelled"
        assert not agent.running

    async def test_the_console_offers_to_resume_it(self, seeded_pool: SqliteConnectionPool) -> None:
        agent, _, _ = await _cancelled_turn(seeded_pool)
        resume, abort = _host(agent).actions("s")
        assert resume.label.startswith("↻ Resume Agent from")
        assert resume.command.startswith("/resume ") and resume.variant == "primary"
        assert abort.command.startswith("/abort ") and abort.label.startswith("✕ Abort")

    async def test_resuming_finishes_the_turn_for_the_same_customer(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        agent, llm, _ = await _cancelled_turn(seeded_pool)
        llm.release.set()
        last = agent.runs.latest_in_session("s")
        result = await agent.resume(last.thread_id)
        assert result.status == "completed"
        assert agent.runs.get(last.thread_id).status == "completed"
        assert _host(agent).actions("s") == []
        assert current_customer() is None

    async def test_the_stop_message_says_how_far_the_turn_got(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        """Where it stopped, then the turn's progress as a tree, then Resume on its own line."""
        _, _, state = await _cancelled_turn(seeded_pool)
        lines = state.response.splitlines()
        assert lines[0].startswith("Stopped before ") and lines[0].split()[2].isupper()
        assert any(line.startswith("├─ request:") for line in lines)
        assert lines[-1].startswith("Everything done so far is saved. Choose ↻ Resume")
        assert "✕ Abort" in lines[-1]

    async def test_resume_runs_as_a_console_turn_not_as_command_output(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        from agent_platform.interface.commands.runs import ResumeCommand

        agent, _, _ = await _cancelled_turn(seeded_pool)
        last = agent.runs.latest_in_session("s")
        result = await ResumeCommand(lambda: agent).run([last.thread_id[:8]])
        assert result.effect.resume_thread == last.thread_id
        assert result.lines == [] and "Resumed" not in result.title

    async def test_abort_closes_the_run_keeps_its_record_and_frees_the_console(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        from agent_platform.config import get_settings
        from agent_platform.interface.commands.runs import AbortCommand

        agent, _, _ = await _cancelled_turn(seeded_pool)
        last = agent.runs.latest_in_session("s")
        result = await AbortCommand(lambda: agent).run([last.thread_id[:8]])
        assert result.lines == [get_settings().durability.stop_report.aborted]
        record = agent.runs.get(last.thread_id)
        assert record.status == "aborted" and record.checkpoint_id
        assert _host(agent).actions("s") == []
        assert agent.runs.resumable(10) == []
        again = await AbortCommand(lambda: agent).run([last.thread_id[:8]])
        assert again.lines == [get_settings().durability.stop_report.nothing_to_abort]
