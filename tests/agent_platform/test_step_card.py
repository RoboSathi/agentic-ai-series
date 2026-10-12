"""One layout for every step card: the configured sections, in order, empty ones left out.

A step that used a model drew two cards, and each step laid its card out its own way. The step
card is one card per step with the same six sections everywhere.
"""

import json

from agent_platform.config import get_settings
from agent_platform.llm.types import CompletionResult, TokenUsage
from agent_platform.observability.step_card import ModelCall, StepCard

_PRIMARY = "nvidia/primary-model"


def _result(content: str, reasoning: str = "", model: str = _PRIMARY) -> CompletionResult:
    return CompletionResult(
        content=content,
        reasoning=reasoning,
        model=model,
        usage=TokenUsage(prompt_tokens=100, completion_tokens=43, total_tokens=143),
    )


class TestTheSections:
    def test_they_follow_the_configured_order_whatever_order_they_were_filled(self) -> None:
        call = ModelCall.of("reflector", _result('{"follow_ups": []}', "thinking"), 1.5, _PRIMARY)
        card = StepCard().evidence("3 claims").decision("stop").model(call).asked("why?")
        titles = list(get_settings().observability.step_card.sections.values())
        assert list(card.fields()) == titles

    def test_an_empty_section_is_left_out(self) -> None:
        card = StepCard().asked("why?").decision("stop")
        assert list(card.fields()) == [
            get_settings().observability.step_card.sections[key] for key in ("asked", "decision")
        ]


class TestTheModelCall:
    def test_the_model_line_names_the_role_in_capitals_and_the_model(self) -> None:
        call = ModelCall.of("reflector", _result("{}"), 4.2, _PRIMARY)
        [line] = (
            StepCard()
            .model(call)
            .fields()[get_settings().observability.step_card.sections["model"]]
        )
        assert str(line).startswith("REFLECTOR · nvidia/primary-model · 143 tokens · 4.2s")

    def test_a_backup_that_answered_is_named(self) -> None:
        call = ModelCall.of("reflector", _result("{}", model="openai/backup"), 1.0, _PRIMARY)
        assert call.backup == "openai/backup"

    def test_without_provider_reasoning_the_answers_own_rationale_is_shown(self) -> None:
        content = json.dumps({"follow_ups": [], "rationale": "R7 needs data after 2024"})
        assert ModelCall.of("reflector", _result(content), 1.0).reasoning == (
            "R7 needs data after 2024"
        )

    def test_the_response_is_a_tree_not_json(self) -> None:
        answer = {"follow_ups": [{"question": "q", "queries": ["a"]}], "rationale": "x"}
        call = ModelCall.of("reflector", _result(json.dumps(answer)), 1.0)
        lines = (
            StepCard()
            .model(call)
            .fields()[get_settings().observability.step_card.sections["response"]]
        )
        assert not any("{" in str(line) for line in lines)
        # The rationale is the Reasoning section; the Response does not repeat it.
        assert not any("rationale" in str(line) for line in lines)


class TestSeveralCallsAndALongAnswer:
    def test_combined_sums_the_calls_and_keeps_the_last_answer(self) -> None:
        first = ModelCall.of("planner", _result('{"a": 1}'), 1.0)
        second = ModelCall.of("planner", _result('{"b": 2}'), 2.0)
        both = ModelCall.combined([first, second])
        assert both is not None
        assert (both.calls, both.tokens, both.seconds) == (2, 286, 3.0)
        assert both.response == '{"b": 2}'
        assert ModelCall.combined([]) is None

    def test_a_long_answer_is_left_out_for_the_steps_own_summary(self) -> None:
        call = ModelCall.of("synthesizer", _result('{"report": "very long"}'), 1.0)
        sections = get_settings().observability.step_card.sections
        fields = StepCard().model(call, response=False).response("Overview · 4 citations").fields()
        assert fields[sections["response"]] == ["Overview · 4 citations"]
