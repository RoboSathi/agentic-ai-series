"""The research agent's profile and role procedures (ADR 0010, ra-0021).

The profile goes below each profile role's `ROLE:` line, with the
procedure that role's signal selects; the judge and the extractor are
left narrow. Scripted roles, the real composition root, real files.
"""

import pytest

from agent_platform.config import get_settings
from agent_platform.guardrails.guards import PydanticValidator
from agent_platform.llm.types import Message
from agent_platform.memory import ProceduralStore
from agent_platform.persistence import SqliteConnectionPool
from agent_platform.skills import build_agent_instructions
from research_agent.config import get_agent_settings
from research_agent.llm.structured import StructuredCaller
from research_agent.prompts.base import RolePrompt, payload_of, role_of
from research_agent.prompts.planner import PlanAnswer
from tests.research_agent.conftest import BuildHarness, ScriptedRoles

ANSWERED = {"direct_answer": True}
_PROFILE = "# Research Agent"


class TestThePreamble:
    def test_the_role_line_stays_first(self) -> None:
        prompt = RolePrompt(role="planner", instructions="Plan.", payload={"a": 1})
        messages = prompt.messages(PlanAnswer, preamble="# AgentProfile\nBe careful.")
        assert role_of(messages) == "planner"
        assert payload_of(messages) == {"a": 1}
        assert messages[0].content.splitlines()[1] == "# AgentProfile"

    def test_no_preamble_changes_nothing(self) -> None:
        prompt = RolePrompt(role="planner", instructions="Plan.", payload={})
        assert prompt.messages(PlanAnswer) == prompt.messages(PlanAnswer, preamble="")


def _system(messages: list[Message]) -> str:
    return str(messages[0].content)


async def test_the_answerer_is_prompted_with_the_profile(harness: BuildHarness) -> None:
    h = harness({"answerer": [{"content": "Paris is the capital of France."}]}, ANSWERED)
    await h.agent.handle("what is the capital of France", "s")
    assert _system(h.roles.models["answerer"].requests[0].messages).startswith(_PROFILE)


async def _sent(role: str, pool: SqliteConnectionPool) -> str:
    roles = ScriptedRoles({role: [{"nodes": []}]})
    instructions = build_agent_instructions(
        get_settings().skills, store=ProceduralStore(pool), audit=None, reporter=None
    )
    caller = StructuredCaller(
        roles,
        PydanticValidator(),
        instructions=instructions,
        profile=get_agent_settings().instructions,
    )
    await caller.call(RolePrompt(role=role, instructions="Do it.", payload={}), PlanAnswer)
    return _system(roles.models[role].requests[0].messages)


@pytest.mark.usefixtures("ra_config")
async def test_a_planner_call_carries_its_procedure(pool: SqliteConnectionPool) -> None:
    system = await _sent("planner", pool)
    assert system.startswith("ROLE: planner\n" + _PROFILE)
    assert "# Procedure: research-planning" in system


@pytest.mark.usefixtures("ra_config")
async def test_the_judge_is_left_narrow(pool: SqliteConnectionPool) -> None:
    system = await _sent("judge", pool)
    assert _PROFILE not in system
    assert "# Procedure:" not in system
