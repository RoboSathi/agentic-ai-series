"""The ops agent's profile and its one procedure (ADR 0010)."""

from typing import Any

from agent_platform.config import get_settings
from tests.ops_agent.conftest import Harness

_ANSWER = "No query needed."


def _system(run: Harness) -> str:
    return str(run.llm.requests[0].messages[0].content)


async def test_every_question_carries_the_profile_and_the_procedure(harness: Any) -> None:
    run: Harness = harness([{"content": _ANSWER}])
    await run.agent.handle("how many denies were there?", "s1")
    system = _system(run)
    assert system.startswith("# Ops Analyst")
    assert "# Procedure: analytics-investigation" in system


async def test_the_activation_is_audited(harness: Any) -> None:
    run: Harness = harness([{"content": _ANSWER}])
    await run.agent.handle("how many denies were there?", "s1")
    action = get_settings().skills.audit.activation_action
    rows = run.rows(
        "SELECT json_extract_string(authority_context, '$.procedures[0]') "
        f"FROM platform_audit_entries WHERE action = '{action}'"
    )
    assert rows == [("analytics-investigation@1",)]


async def test_without_procedural_memory_only_the_profile(harness: Any) -> None:
    run: Harness = harness([{"content": _ANSWER}], features={"memory.procedural": False})
    await run.agent.handle("how many denies were there?", "s1")
    assert "# Procedure:" not in _system(run)
    assert run.agent.skill_catalog is None
