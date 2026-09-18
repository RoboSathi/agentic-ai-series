"""The live status feed's labels, widgets and clipboard (LLD §20)."""

from pathlib import Path

from agentic_commons.observability.types import StatusEvent
from agentic_commons.tui import (
    Osc52ClipboardService,
    StageLabelRegistry,
    StatusFeedBuffer,
    StatusLineWidget,
    TokenCounterWidget,
)
from agentic_commons.tui.widgets import _compact, _elapsed

# Every stage key the harness emits this phase. The contract test below
# fails if one of these has no configured label — a silent fallback would
# hide the omission rather than surface it.
EMITTED_STAGES = {
    "prompt_injection_check",
    "context_assembly",
    "llm_call",
    "llm_response_received",
    "gatekeeper_evaluation",
    "tool_execution",
    "finalize",
    "intake",
}


class TestStageLabels:
    def test_every_emitted_stage_has_a_label(self) -> None:
        """The contract test — a missing entry fails here, not silently."""
        assert StageLabelRegistry().missing(EMITTED_STAGES) == set()

    def test_labels_come_from_config(self) -> None:
        assert StageLabelRegistry().label_for("llm_call") == "Calling model…"

    def test_a_tool_step_reads_as_work_not_as_a_method_name(self) -> None:
        """The person watching cares what the agent is doing."""
        label = StageLabelRegistry().label_for("tool_execution", tool_name="process_refund")
        assert label == "Issuing the refund…"
        assert "process_refund" not in label

    def test_a_namespaced_tool_resolves_to_the_same_wording(self) -> None:
        registry = StageLabelRegistry()
        assert registry.tool_label("support.process_refund") == registry.tool_label(
            "process_refund"
        )

    def test_an_unlabelled_tool_degrades_readably(self) -> None:
        assert StageLabelRegistry().tool_label("send_carrier_email") == (
            "Running send carrier email…"
        )

    def test_steps_are_not_all_one_colour(self) -> None:
        """A feed that is entirely green communicates nothing."""
        registry = StageLabelRegistry()
        styles = {
            registry.style_for(stage)
            for stage in ("prompt_injection_check", "context_assembly", "llm_call", "finalize")
        }
        assert len(styles) >= 3

    def test_a_refused_action_is_coloured_as_a_problem(self) -> None:
        assert StageLabelRegistry().style_for("tool_refused") == "red"

    def test_a_suppressed_repeat_is_muted(self) -> None:
        """It is not work done, so it must not look like work done."""
        assert StageLabelRegistry().style_for("tool_suppressed") == "bright_black"

    def test_a_missing_substitution_leaves_the_template_readable(self) -> None:
        assert StageLabelRegistry().label_for("tool_execution") != ""

    def test_an_unknown_stage_degrades_readably(self) -> None:
        assert StageLabelRegistry().label_for("some_new_stage") == "Some new stage"


class TestTokenCounter:
    def test_tracks_turn_and_session_separately(self) -> None:
        """They answer different questions."""
        counter = TokenCounterWidget()
        counter.add(100)
        counter.start_turn()
        counter.add(50)
        assert counter.turn_tokens == 50 and counter.session_tokens == 150

    def test_renders_both_counts(self) -> None:
        counter = TokenCounterWidget()
        counter.add(1200)
        rendered = counter.render().plain
        assert "1.2k turn" in rendered and "1.2k session" in rendered


class TestStatusLine:
    def test_shows_the_stage_and_a_timer(self) -> None:
        line = StatusLineWidget()
        line.begin("Assembling context…")
        assert "Assembling context…" in line.render().plain

    def test_is_empty_before_anything_starts(self) -> None:
        assert StatusLineWidget().render().plain == ""

    def test_the_spinner_advances_while_running(self) -> None:
        line = StatusLineWidget()
        line.begin("working")
        first = line.render().plain[0]
        line.tick()
        assert line.render().plain[0] != first

    def test_the_spinner_stops_when_finished(self) -> None:
        line = StatusLineWidget()
        line.begin("working")
        line.finish()
        assert line.render().plain.startswith("•")


class TestStatusFeedBuffer:
    def test_renders_a_completed_stage_with_a_check(self) -> None:
        panel = StatusFeedBuffer()
        panel.show_status(StatusEvent(stage="llm_call", status="completed"))
        assert panel.as_text().startswith("✓")

    def test_renders_a_failure_with_its_detail(self) -> None:
        panel = StatusFeedBuffer()
        panel.show_status(StatusEvent(stage="llm_call", status="failed", detail="timeout"))
        assert "✗" in panel.as_text() and "timeout" in panel.as_text()

    def test_started_events_do_not_add_a_line(self) -> None:
        """They drive the status line instead; a line per start would double up."""
        panel = StatusFeedBuffer()
        panel.show_status(StatusEvent(stage="llm_call", status="started"))
        assert panel.as_text() == ""

    def test_token_deltas_are_shown(self) -> None:
        panel = StatusFeedBuffer()
        panel.show_status(
            StatusEvent(stage="llm_response_received", status="completed", tokens_delta=2400)
        )
        assert "2.4k" in panel.as_text()

    def test_the_backlog_is_bounded(self) -> None:
        panel = StatusFeedBuffer()
        for index in range(600):
            panel.note(str(index))
        assert len(panel.as_text().splitlines()) <= 500

    def test_clearing_empties_the_panel(self) -> None:
        panel = StatusFeedBuffer()
        panel.note("something")
        panel.clear()
        assert panel.as_text() == ""


class TestClipboard:
    def test_always_writes_an_export_file(self, tmp_path: Path) -> None:
        """A copy that silently does nothing is worse than one that refuses."""
        path = Osc52ClipboardService(tmp_path).copy("feed contents")
        assert Path(path).read_text() == "feed contents"

    def test_a_large_payload_still_exports(self, tmp_path: Path) -> None:
        """OSC 52 has a size limit; the file does not."""
        payload = "x" * 200_000
        path = Osc52ClipboardService(tmp_path).copy(payload)
        assert len(Path(path).read_text()) == 200_000


class TestFormatting:
    def test_compacts_large_token_counts(self) -> None:
        assert _compact(58_800) == "58.8k" and _compact(42) == "42"

    def test_formats_elapsed_time_by_magnitude(self) -> None:
        assert _elapsed(4) == "4s"
        assert _elapsed(64) == "1m 04s"
        assert _elapsed(3_720) == "1h 02m"
