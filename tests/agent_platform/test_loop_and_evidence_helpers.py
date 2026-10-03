"""Shared pieces every agent's reasoning loop uses: claim splitting, tool evidence, step cards."""

from typing import Any

from agent_platform.config import get_settings
from agent_platform.guardrails.response import _surviving_sentences
from agent_platform.hallucination_protection import (
    NOT_EXECUTED_MARKER,
    ClaimDecomposer,
    Observation,
    plain_text,
    sentence_pieces,
    split_sentences,
    tool_evidence,
)
from agent_platform.hallucination_protection.types import DetectionVerdict
from agent_platform.logging import describe_error
from agent_platform.observability import InMemoryLiveStatusFeed, StageReporter
from agent_platform.observability.loop_reporter import LOOP_STEP_FIELD, LoopReporter, mark_step
from agent_platform.observability.tool_fields import display_tool_name, tool_request_fields

ANSWER = "Tomorrow in Pune:\n\n- **Temperature**: 22.7 °C – 32.5 °C\n- Wind 2.8 km/h. Humid."


class TestClaims:
    def test_a_line_break_ends_a_claim_as_a_full_stop_does(self) -> None:
        assert split_sentences(ANSWER) == [
            "Tomorrow in Pune:",
            "- **Temperature**: 22.7 °C – 32.5 °C",
            "- Wind 2.8 km/h.",
            "Humid.",
        ]

    def test_pieces_rebuild_the_text_exactly(self) -> None:
        assert "".join(s + sep for s, sep in sentence_pieces(ANSWER)) == ANSWER

    def test_markdown_is_layout_not_content(self) -> None:
        assert plain_text("- **Temperature**: 22.7 °C") == "Temperature: 22.7 °C"

    def test_a_lead_in_line_claims_nothing(self) -> None:
        claims = [claim.text for claim in ClaimDecomposer().decompose(ANSWER)]
        assert "Tomorrow in Pune:" not in claims
        assert "Temperature: 22.7 °C – 32.5 °C" in claims

    def test_removing_a_refuted_line_keeps_the_list_a_list(self) -> None:
        verdicts = [_verdict(sentence=2, supported=False)]
        assert _surviving_sentences(ANSWER, verdicts) == (
            "Tomorrow in Pune:\n\n- **Temperature**: 22.7 °C – 32.5 °C\nHumid."
        )


class TestToolEvidence:
    def test_one_premise_per_sentence_in_call_order(self) -> None:
        chunks = tool_evidence(
            [Observation("A is open. B is shut.\nC is late.", source="u1"), Observation("D.")]
        )
        assert [c.content for c in chunks] == ["A is open.", "B is shut.", "C is late.", "D."]
        assert {c.kind for c in chunks} == {"tool_result"}
        assert [c.source for c in chunks][:3] == ["u1"] * 3

    def test_a_refusal_stays_whole(self) -> None:
        line = f"{NOT_EXECUTED_MARKER} refund(order_id=1). Over the limit. Ask a person."
        assert [c.content for c in tool_evidence([Observation(line)])] == [line]


class TestStepCards:
    def test_a_loop_card_is_numbered_and_says_its_cap(self) -> None:
        summary, fields = mark_step(
            "thought", "decided", {"a": 1}, step=2, max_steps=6, stages={"thought"}
        )
        prefix = get_settings().observability.loop_step_prefix.format(step=2)
        assert summary == f"{prefix}decided"
        assert fields is not None and fields[LOOP_STEP_FIELD].startswith("2 of 6")

    def test_a_card_outside_the_loop_is_left_alone(self) -> None:
        assert mark_step("answer", "done", None, step=2, max_steps=6, stages={"thought"}) == (
            "done",
            None,
        )

    def test_the_reporter_numbers_what_it_emits(self) -> None:
        feed = InMemoryLiveStatusFeed(max_events=0)
        seen: list[Any] = []
        feed.subscribe(seen.append)
        loop = LoopReporter(
            StageReporter(feed, "r"), max_steps=3, stages={"action"}, detail="{step}/{max}"
        )
        loop.begin(1)
        loop.emit("action", summary="weather(Pune)", fields={"tool": "weather"})
        assert seen[0].fields[LOOP_STEP_FIELD] == "1/3"


def test_tool_fields_name_the_tool_and_every_argument() -> None:
    fields = tool_request_fields("answer__weather", '{"location": "Pune"}')
    assert (fields["tool"], fields["arg_location"]) == ("weather", "Pune")
    assert display_tool_name("support.process_refund") == "process_refund"


def test_an_error_reads_as_one_line() -> None:
    assert describe_error(TimeoutError()) == "TimeoutError"
    assert describe_error(ValueError("bad")) == "ValueError: bad"


def _verdict(sentence: int, supported: bool) -> DetectionVerdict:
    return DetectionVerdict(
        claim="x", supported=supported, confidence=0.9, sentence=sentence, undetermined=False
    )
