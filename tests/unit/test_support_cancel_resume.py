"""The support agent's Cancel → Resume, as the console's buttons drive it.

A turn cancelled while the model is answering stops at once, says where it
stopped, and is offered back as "Resume from <node>"; resumed, it finishes
from that node with the customer bound again.
"""

import asyncio
from typing import Any

from agent_platform.llm.types import CompletionRequest, CompletionResult
from agent_platform.persistence import SqliteConnectionPool
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
        [action] = _host(agent).actions("s")
        assert action.label.startswith("↻ Resume from") and action.command.startswith("/resume ")
        assert action.variant == "primary"

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
