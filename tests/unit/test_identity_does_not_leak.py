"""A resumed run's customer ends with the run, and "show orders" is always the table.

Found live (2026-10-03): `/queue approve` resumed a paused turn for customer
#81, and `_rebind` bound #81 into the console's own context with a bare
`set_current_customer`. The next session's first turn found a customer
already bound, logged "Customer already verified", and sent the login code
to the model as an order number. No menu was ever built, so "show orders"
went to the model too and came back as prose.
"""

from typing import Any

from agent_platform.durability.types import RunOutcome, RunRecord
from agent_platform.persistence import SqliteConnectionPool
from customer_support_agent.domain.identity import (
    Customer,
    bound_customer,
    current_customer,
    set_current_customer,
)
from customer_support_agent.graph import build_agent
from customer_support_agent.graph.state import AgentState
from tests.conftest import CUSTOMER_CODE, ScriptedLLM

_ADA = Customer(customer_id=1, name="Ada Lovelace", email="ada@example.com")


class _PausedRun:
    """A runner holding one completed-on-resume thread for session A, customer 1."""

    def __init__(self) -> None:
        self.prior = AgentState(query="refund 10001", session_id="A", customer_id=1)

    async def values(self, thread_id: str) -> dict[str, Any]:
        return self.prior.model_dump()

    async def resume(self, thread_id: str, decision: Any, reporter: Any) -> RunOutcome:
        assert current_customer() == _ADA, "the resumed run itself is the customer's"
        values = self.prior.model_copy(update={"response": "Your refund was approved."})
        return RunOutcome(thread_id=thread_id, status="completed", values=values.model_dump())


class _Runs:
    def get(self, thread_id: str) -> RunRecord:
        return RunRecord(thread_id=thread_id, agent_id="test", session_id="A", status="interrupted")


def _agent(pool: SqliteConnectionPool) -> Any:
    agent = build_agent(llm=ScriptedLLM([{"content": "unused"}]), pool=pool, with_memory=False)
    agent._real = (agent._runner, agent._runs)
    agent._runner = _PausedRun()
    agent._runs = _Runs()
    agent._sessions["A"] = _ADA
    return agent


class TestAResumedRunsIdentityEndsWithIt:
    async def test_the_callers_context_is_unbound_after_a_resume(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        set_current_customer(None)
        await _agent(seeded_pool).resume("A:thread", {"decision": "approved"})
        assert current_customer() is None

    async def test_the_next_session_is_asked_for_its_code(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        set_current_customer(None)
        agent = _agent(seeded_pool)
        await agent.resume("A:thread", {"decision": "approved"})
        agent._runner, agent._runs = agent._real
        state = await agent.handle("7VE2T3T", session_id="B")
        assert "couldn't find an account" in state.response
        assert "B" not in agent._sessions

    def test_a_binding_restores_what_was_bound_before(self) -> None:
        set_current_customer(None)
        with bound_customer(_ADA):
            assert current_customer() == _ADA
        assert current_customer() is None


class TestShowOrdersIsAlwaysTheTable:
    async def test_a_session_verified_upstream_gets_the_table_without_a_model_call(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        set_current_customer(None)
        llm = ScriptedLLM([{"content": "should not be called"}])
        agent = build_agent(llm=llm, pool=seeded_pool, with_memory=False)
        state = await agent.handle("show orders", session_id="up", customer=_ADA)
        assert "10001" in state.response and "10002" in state.response
        assert llm.requests == []
        assert agent._menus["up"].size == 2

    async def test_after_verifying_show_orders_is_the_same_table(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        set_current_customer(None)
        llm = ScriptedLLM([{"content": "should not be called"}])
        agent = build_agent(llm=llm, pool=seeded_pool, with_memory=False)
        await agent.handle(CUSTOMER_CODE, session_id="s")
        state = await agent.handle("show orders", session_id="s")
        assert "10001" in state.response and "10002" in state.response
        assert llm.requests == []
