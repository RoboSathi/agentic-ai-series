"""LOOP CONTROL in the research agent's direct answer: every round says where it goes and why.

The platform card CSA and ops draw (`StepEvents.loop_control`), emitted by the node that
decides, never by a routing edge. Scripted answerer, the offline clock tool, the real graph.
"""

from typing import Any

from agent_platform.llm.types import ModelToolCall
from agent_platform.observability.types import StatusEvent
from research_agent.config import get_agent_settings
from tests.research_agent.conftest import BuildHarness

ANSWERED = {"direct_answer": True}
_LOOP = "loop_control"


def _clock(call_id: str = "c1") -> ModelToolCall:
    return ModelToolCall(
        id=call_id, name="answer__current_datetime", arguments='{"timezone": "UTC"}'
    )


async def _cards(harness: BuildHarness, script: list[dict[str, Any]]) -> list[StatusEvent]:
    h = harness({"answerer": script}, ANSWERED)
    events: list[StatusEvent] = []
    h.feed.subscribe(events.append)
    await h.agent.handle("what time is it in UTC?", "s")
    return [e for e in events if e.stage == _LOOP]


def _next(card: StatusEvent) -> str:
    return str(card.fields["next_step"])


async def test_each_round_says_where_it_goes(harness: BuildHarness) -> None:
    cards = await _cards(harness, [{"tool_calls": [_clock()]}, {"content": "It is noon UTC."}])
    words = get_agent_settings().answer.messages.loop_next
    assert [_next(card) for card in cards] == [words["act"], words["reason"], words["ground"]]


async def test_the_card_carries_every_budget_it_weighed(harness: BuildHarness) -> None:
    [first, *_] = await _cards(harness, [{"tool_calls": [_clock()]}, {"content": "Noon."}])
    criteria = first.fields["criteria"]
    assert {"steps", "tool_calls", "tokens_left", "seconds_left"} <= set(criteria)
    assert criteria["model_requested_tools"] is True


async def test_tool_calls_cut_to_the_budget_are_counted(harness: BuildHarness) -> None:
    cap = get_agent_settings().answer.budget.max_tool_calls
    calls = [_clock(f"c{index}") for index in range(cap + 2)]
    [first, *_] = await _cards(harness, [{"tool_calls": calls}, {"content": "Noon."}])
    assert first.fields["criteria"]["dropped_tool_calls"] == 2
