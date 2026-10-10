"""An LLM RESPONSE card names what the call produced, not what it was hoped to produce.

The support agent's loop step was labelled `tool_selection` before the call, so a step that
answered (the turn's final reply) read "LLM response · tool_selection".
"""

from agent_platform.config import get_settings
from agent_platform.llm.types import CompletionResult, ModelToolCall
from agent_platform.observability import InMemoryLiveStatusFeed, StageReporter
from agent_platform.observability.step_events import LLM_RESPONSE, StepEvents
from agent_platform.observability.types import StatusEvent


def _response(result: CompletionResult, purpose: str) -> StatusEvent:
    feed = InMemoryLiveStatusFeed(max_events=0)
    events: list[StatusEvent] = []
    feed.subscribe(events.append)
    StepEvents(StageReporter(feed, run_id="t"), max_steps=5).llm_response(result, purpose=purpose)
    return next(event for event in events if event.stage == LLM_RESPONSE)


def test_a_next_action_call_that_answered_is_a_reply() -> None:
    words = get_settings().observability.llm_purposes
    event = _response(CompletionResult(content="Your refund was issued."), words.next_action)
    assert event.fields["purpose"] == words.reply
    assert (event.summary or "").endswith(words.reply)


def test_a_next_action_call_that_asked_for_a_tool_is_tool_selection() -> None:
    words = get_settings().observability.llm_purposes
    call = ModelToolCall(id="c1", name="support__get_order_status", arguments="{}")
    event = _response(CompletionResult(tool_calls=[call]), words.next_action)
    assert event.fields["purpose"] == words.tool_selection


def test_any_other_purpose_is_kept_as_given() -> None:
    words = get_settings().observability.llm_purposes
    event = _response(CompletionResult(content="Rewritten."), words.reply_revision)
    assert event.fields["purpose"] == words.reply_revision
