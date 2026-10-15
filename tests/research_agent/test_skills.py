"""The research agent's profile and learned skills (ADR 0010, ra-0021, ra-0023).

The profile goes on the direct answer only. Every run role's own
instructions already are its skill, so a role call carries no profile
and no skill — a second copy cost tokens on every call.
"""

from types import SimpleNamespace

import pytest

from agent_platform.config import get_settings
from agent_platform.core.persistence import SqliteConnectionPool
from agent_platform.guardrails.guards import PydanticValidator
from agent_platform.llm.types import Message
from agent_platform.memory import MemoryFact, Provenance, SkillStore
from agent_platform.memory.procedural.skills import build_agent_instructions
from research_agent.config import get_agent_settings
from research_agent.graph.run_nodes import RunNodes
from research_agent.llm.structured import StructuredCaller
from research_agent.prompts import planner as planner_prompt
from research_agent.prompts.base import SCHEMA_INTRO, RolePrompt, role_of
from research_agent.prompts.planner import PlanAnswer
from tests.research_agent.conftest import BuildHarness, ScriptedRoles, payload_of
from tests.research_agent.test_reflection_and_cards import _contract

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
    # The role's instructions, then its answer schema: no profile, no skills.
    assert system.startswith(f"ROLE: {role}\nDo it.{SCHEMA_INTRO}")


@pytest.mark.usefixtures("ra_config")
def test_procedural_memory_holds_learned_skills_only() -> None:
    """ra-0023: composed again, but no role has an authored skill (ra-0021 stands)."""
    assert get_settings().features.is_on("memory.procedural")
    assert not get_settings().skills.resolved_skills_dir.joinpath("planner").exists()


def _learned(scope: str) -> str:
    return (
        "---\nname: learned-criterion-failed-x\ndescription: Learned from 3 runs - cite sources.\n"
        f"triggers: [{scope}]\nkeywords: [sources]\n---\n## Preconditions\n- p\n\n"
        "## Procedure\n1. Cite primary sources.\n\n## Boundaries\n- b\n\n## Completion\nDone.\n"
    )


def _nodes(scope_template: str) -> RunNodes:
    settings = get_settings()
    store = SkillStore()
    scope = scope_template.format(archetype="explainer")
    store.write(
        MemoryFact(
            key="learned-criterion-failed-x",
            value=_learned(scope),
            category=settings.memory.dream.learned_skill.artifact_type,
            provenance=Provenance(source="curator"),
        ),
        Provenance(source="curator"),
    )
    instructions = build_agent_instructions(settings.skills, store=store, audit=None, reporter=None)
    parts = SimpleNamespace(skills=instructions, skill_scope=scope_template)
    return RunNodes(parts)  # type: ignore[arg-type]


@pytest.mark.usefixtures("ra_config")
class TestLearnedSkillsAtPlan:
    def test_a_skill_learned_for_this_kind_reaches_the_planner(self) -> None:
        nodes = _nodes(get_agent_settings().memory.scope)
        contract = SimpleNamespace(brief=SimpleNamespace(archetype="explainer", topic="t"))
        [skill] = nodes._learned_skills(contract)
        assert "Cite primary sources." in skill
        request = planner_prompt.build(_contract(), 4, 2, [skill])
        assert payload_of(request.messages(PlanAnswer))["skills_learned_from_past_runs"] == [skill]

    def test_another_kind_gets_none(self) -> None:
        nodes = _nodes(get_agent_settings().memory.scope)
        contract = SimpleNamespace(brief=SimpleNamespace(archetype="comparison", topic="t"))
        assert nodes._learned_skills(contract) == []
