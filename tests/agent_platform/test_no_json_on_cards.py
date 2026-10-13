"""No card shows JSON: tool arguments read as `key=value`, a tool's JSON answer as a tree."""

import json

from rich.text import Text

from agent_platform.interface.presentation.text_tree import (
    inline_arguments,
    message_lines,
    readable_lines,
)
from agent_platform.interface.tui.widgets import _append_grounding, _block_lines, _narrative_lines
from agent_platform.llm.prompt import manifest_from_messages
from agent_platform.llm.types import Message, ModelToolCall
from agent_platform.observability.context_card import context_fields

ARGS = json.dumps({"sql": "SELECT 1", "limit": 5})
RESULT = json.dumps({"columns": ["n"], "rows": [[1]], "notes": []})


def _no_json(lines: list[str]) -> None:
    text = "\n".join(lines)
    assert '{"' not in text and '":' not in text, text


def test_arguments_read_as_key_value() -> None:
    assert inline_arguments(ARGS) == "sql=SELECT 1, limit=5"
    assert inline_arguments({"by": {"day": True}}) == "by={day=true}"
    assert inline_arguments("") == ""


def test_a_json_answer_reads_as_a_tree() -> None:
    lines = readable_lines(RESULT)
    assert lines[0].startswith("├─ columns")
    _no_json(lines)
    assert readable_lines("plain\ntext") == ["plain", "text"]


def test_a_message_with_calls_and_a_json_result() -> None:
    call = message_lines("assistant", "", [("run_query", ARGS)])
    assert call == ["assistant:", "  → run_query(sql=SELECT 1, limit=5)"]
    result = message_lines("tool (run_query)", RESULT, [])
    assert result[0] == "tool (run_query):"
    _no_json(result)


def test_the_llm_request_transcript_shows_no_json() -> None:
    transcript = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"tool": "run_query", "arguments": ARGS}],
        },
        {"role": "tool", "tool": "run_query", "content": RESULT},
    ]
    _no_json(_block_lines(transcript))


def test_the_context_card_sections_show_no_json() -> None:
    messages = [
        Message(role="user", content="How many?"),
        Message(
            role="assistant",
            content="",
            tool_calls=[ModelToolCall(id="c1", name="run_query", arguments=ARGS)],
        ),
        Message(role="tool", content=RESULT, tool_call_id="c1"),
    ]
    fields = context_fields(manifest_from_messages(messages, None), "m")
    sections = [lines for title, lines in fields.items() if "tokens" in title]
    assert sections
    for lines in sections:
        _no_json([str(line) for line in lines])


def test_a_nested_narrative_section_is_a_tree() -> None:
    lines = [text for text, _ in _narrative_lines({"server": {"name": "a", "tools": ["x"]}})]
    assert lines[0] == "└─ server:"
    _no_json(lines)


def test_the_grounding_card_shows_level_two_beside_the_confidence_block() -> None:
    body = Text()
    _append_grounding(
        body,
        {
            "outcome": "PASSED · reply sent unchanged",
            "confidence": {"available": True, "model": "m", "sentences": []},
            "levels": [
                {"level": 1, "name": "token confidence", "verdict": "PASS", "summary": "l1"},
                {
                    "level": 2,
                    "name": "figures in query results",
                    "verdict": "PASS",
                    "summary": "l2",
                },
            ],
        },
    )
    shown = body.plain
    assert "logprobs · m" in shown
    assert "level 2 · figures in query results" in shown
    # Level 1 is the confidence block, not a second row.
    assert "level 1 ·" not in shown
