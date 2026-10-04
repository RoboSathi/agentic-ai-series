"""The research run's Stop pauses it; Resume continues with its state restored; Abort ends it.

The durable runner is a fake here: what is under test is the agent's side — the stop report it
writes, the ledger and budget it restores before resuming, and the phase each choice leaves.
The runner's own pause/abort is tested with the platform.
"""

from typing import Any

import pytest

from agent_platform.config import get_settings
from agent_platform.durability.types import RunOutcome, RunRecord
from research_agent.config import get_agent_settings
from tests.research_agent.conftest import BuildHarness

_THREAD = "c"


def _contract() -> Any:
    from research_agent.domain.rubric import Contract, Rubric
    from research_agent.domain.scope import ScopeBrief

    brief = ScopeBrief(
        archetype="explainer",
        topic="agent memory",
        budget=get_agent_settings().budget,
        concepts=("x",),
    )
    return Contract.freeze(_THREAD, brief, Rubric(), brief.budget)


class _Registry:
    def get(self, thread_id: str) -> RunRecord:
        return RunRecord(thread_id=thread_id, agent_id="a", session_id="s", status="cancelled")


class _Runs:
    """A durable runner that stops the first run and finishes the resume."""

    registry = _Registry()

    def __init__(self, contract: Any) -> None:
        self._values = {"contract": contract, "round": 2, "spent": {"searches": 3.0}}
        self.resumed = False

    async def start(self, *_: Any, **__: Any) -> RunOutcome:
        return RunOutcome(
            thread_id=_THREAD, status="cancelled", values=self._values, next_nodes=["reflect"]
        )

    async def values(self, thread_id: str) -> dict[str, Any]:
        return self._values

    async def resume(self, *_: Any) -> RunOutcome:
        self.resumed = True
        done = {**self._values, "status": "CANCELLED", "stop_reason": "operator"}
        return RunOutcome(thread_id=_THREAD, status="completed", values=done)

    async def abort(self, thread_id: str, reporter: Any = None) -> RunOutcome:
        return RunOutcome(thread_id=thread_id, status="aborted")


@pytest.fixture
def stopped(harness: BuildHarness) -> Any:
    agent = harness().agent
    agent._runs = _Runs(_contract())
    return agent


async def test_a_stopped_run_reports_where_and_how_far_then_the_choices(stopped: Any) -> None:
    result = await stopped._run(_contract(), "s", stopped.session("s"))
    lines = result.response.splitlines()
    assert result.phase == "STOPPED" and lines[0].startswith("Stopped before REFLECT")
    assert any("agent memory" in line for line in lines)
    assert "✕ Abort" in lines[-1]


async def test_resume_restores_the_budget_spent_before_it_continues(stopped: Any) -> None:
    await stopped._run(_contract(), "s", stopped.session("s"))
    await stopped.resume(_THREAD)
    assert stopped._runs.resumed
    assert stopped._nodes.resources.budget.snapshot()["searches"].spent == 3.0


async def test_abort_ends_the_run_and_frees_the_session(stopped: Any) -> None:
    await stopped._run(_contract(), "s", stopped.session("s"))
    result = await stopped.abort(_THREAD)
    assert result.response == get_settings().durability.stop_report.aborted
    assert stopped.session("s").phase == "CANCELLED"
