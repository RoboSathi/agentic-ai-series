"""The research agent's profile (ADR 0010, ra-0021).

The profile goes on the direct answer only. Every run role's own
instructions already are its procedure, so a role call carries no profile
and no procedure — a second copy cost tokens on every call.
"""

import pytest

from agent_platform.config import get_settings
from agent_platform.guardrails.guards import PydanticValidator
from agent_platform.llm.types import Message
from agent_platform.persistence import SqliteConnectionPool
from agent_platform.skills import build_agent_instructions
from research_agent.config import get_agent_settings
from research_agent.llm.structured import StructuredCaller
from research_agent.prompts.base import RolePrompt, role_of
from research_agent.prompts.planner import PlanAnswer
from tests.research_agent.conftest import BuildHarness, ScriptedRoles, payload_of

ANSWERED = {"direct_answer": True}
_PROFILE = "# Research Agent"


class TestThePreamble:
    def test_the_role_line_stays_first(self) -> None:
        prompt = RolePrompt(role="planner", instructions="Plan.", payload={"a": 1})
        messages = prompt.messages(PlanAnswer, preamble="# Profile\nBe careful.")
        assert role_of(messages) == "planner"
        assert payload_of(messages) == {"a": 1}
        assert messages[0].content.splitlines()[1] == "# Profile"

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
        get_settings().skills, store=None, audit=None, reporter=None
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
@pytest.mark.parametrize("role", ["planner", "reflector", "synthesizer", "judge", "extractor"])
async def test_a_run_role_carries_its_own_instructions_only(
    role: str, pool: SqliteConnectionPool
) -> None:
    system = await _sent(role, pool)
    assert system == f"ROLE: {role}\nDo it."


@pytest.mark.usefixtures("ra_config")
def test_procedural_memory_is_not_composed() -> None:
    assert not get_settings().features.is_on("memory.procedural")
