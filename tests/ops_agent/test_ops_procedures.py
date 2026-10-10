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


async def test_the_procedure_prefetches_the_catalogue_and_the_views(harness: Any) -> None:
    """The model starts with what its first two calls used to fetch."""
    run: Harness = harness([{"content": _ANSWER}])
    await run.agent.handle("how many denies were there?", "s1")
    first = run.llm.requests[0].messages
    asked = [call.name for message in first if message.tool_calls for call in message.tool_calls]
    assert asked == ["list_metrics", "list_views"]
    results = [m for m in first if m.role == "tool"]
    assert len(results) == 2 and all(result.content for result in results)


async def test_the_procedure_card_names_what_was_prefetched(harness: Any) -> None:
    run: Harness = harness([{"content": _ANSWER}])
    events: list[Any] = []
    run.feed.subscribe(events.append)
    await run.agent.handle("how many denies were there?", "s1")
    stage = get_settings().skills.observability.activation_stage
    [card] = [e for e in events if e.stage == stage]
    assert card.summary.startswith("analytics-investigation v1 · always used")
    assert card.summary.endswith("reads first: list_metrics, list_views")


async def test_the_card_lists_exactly_the_tools_the_model_was_offered(harness: Any) -> None:
    run: Harness = harness([{"content": _ANSWER}])
    events: list[Any] = []
    run.feed.subscribe(events.append)
    await run.agent.handle("how many denies were there?", "s1")
    stage = get_settings().skills.observability.activation_stage
    [card] = [e for e in events if e.stage == stage]
    sent = [tool["function"]["name"] for tool in run.llm.requests[0].tools or []]
    assert sorted(card.fields["tools_offered"]) == sorted(sent)


async def test_the_card_comes_before_the_reads_it_explains(harness: Any) -> None:
    run: Harness = harness([{"content": _ANSWER}])
    events: list[Any] = []
    run.feed.subscribe(events.append)
    await run.agent.handle("how many denies were there?", "s1")
    stage = get_settings().skills.observability.activation_stage
    order = [e.stage for e in events if e.status != "started"]
    assert order.index(stage) < order.index("tool_request")
