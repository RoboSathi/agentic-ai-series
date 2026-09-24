"""The live status feed's labels, widgets and clipboard (LLD §20)."""

from pathlib import Path
from types import SimpleNamespace

from agentic_commons.commands import ConsoleEffect
from agentic_commons.commands.harness import HarnessCommand
from agentic_commons.llm.types import Message
from agentic_commons.persistence import SqliteConnectionPool
from agentic_commons.tui import (
    Osc52ClipboardService,
    StageLabelRegistry,
    StatusLineWidget,
    TokenCounterWidget,
)
from agentic_commons.tui.app import AgenticTuiApp
from agentic_commons.tui.widgets import _compact, _elapsed
from customer_support_agent.domain.identity import Customer
from customer_support_agent.graph import build_agent
from tests.conftest import ScriptedLLM

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

    def test_the_counter_draws_nothing_itself(self) -> None:
        """The bottom line shows elapsed time and tokens, and nothing else.

        The counter owns the totals; the status line above it is where a
        reader already looks, and showing them twice is noise.
        """
        counter = TokenCounterWidget()
        counter.add(1200)
        assert counter.render().plain == ""

    def test_a_new_turn_resets_the_turn_counters_only(self) -> None:
        counter = TokenCounterWidget()
        counter.add(1000)
        counter.start_turn()
        counter.add(500)
        assert counter.turn_tokens == 500 and counter.session_tokens == 1500


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


class TestLayout:
    """The prompt sits under the transcript, with its status below it.

    Reported as "the chat window is too low": `dock: bottom` pins a
    widget to the screen edge and removes it from the flow, which left
    the prompt stranded at the very bottom. Nothing is docked now.
    """

    async def test_the_prompt_is_below_the_transcript(self) -> None:
        from textual.containers import VerticalScroll
        from textual.widgets import Input

        from agentic_commons.tui.app import AgenticTuiApp

        app = AgenticTuiApp()
        async with app.run_test(size=(78, 26)):
            feed = app.query_one("#feed", VerticalScroll)
            prompt = app.query_one("#query", Input)
            assert prompt.region.y > feed.region.y

    async def test_the_status_line_sits_under_the_prompt(self) -> None:
        """Where it reads as a status on the work, as in Claude Code."""
        from textual.widgets import Input

        from agentic_commons.tui.app import AgenticTuiApp

        app = AgenticTuiApp()
        async with app.run_test(size=(78, 26)):
            prompt = app.query_one("#query", Input)
            assert app._status.region.y >= prompt.region.y + prompt.region.height

    async def test_nothing_is_docked(self) -> None:
        """Three docked widgets once landed in the same place."""
        from agentic_commons.tui.app import AgenticTuiApp

        assert "dock:" not in AgenticTuiApp.CSS.replace(" ", "")

    async def test_the_transcript_takes_the_remaining_height(self) -> None:
        from textual.containers import VerticalScroll

        from agentic_commons.tui.app import AgenticTuiApp

        app = AgenticTuiApp()
        async with app.run_test(size=(78, 26)):
            assert app.query_one("#feed", VerticalScroll).region.height > 10

    async def test_one_session_spans_the_whole_window(self) -> None:
        """Keying per turn would give every message its own conversation."""
        from agentic_commons.tui.app import AgenticTuiApp

        app = AgenticTuiApp()
        first = app._session_id
        app._turn += 1
        assert app._session_id == first


class TestTheAppRebuildsItsAgent:
    """`/harness` changes which subsystems are enabled, which changes what
    the composition root would wire — and the agent already on screen was
    wired under the previous answer."""

    def _app(self) -> AgenticTuiApp:
        built: list[object] = []

        def factory(reporter: object) -> object:
            built.append(reporter)
            return SimpleNamespace(memory=None, handle=None, _built=len(built))

        return AgenticTuiApp(factory)

    def test_a_rebuild_effect_replaces_the_agent(self) -> None:
        app = self._app()
        before = app._agent
        app._apply(ConsoleEffect(rebuild_agent=True))
        assert app._agent is not before

    def test_other_effects_leave_the_agent_alone(self) -> None:
        app = self._app()
        before = app._agent
        app._apply(ConsoleEffect(clear_transcript=True))
        assert app._agent is before

    def test_an_app_with_no_factory_survives_a_rebuild(self) -> None:
        """The feed-only console has nothing to rebuild and must not crash."""
        app = AgenticTuiApp()
        app._apply(ConsoleEffect(rebuild_agent=True))
        assert app._agent is None

    def test_an_episode_names_the_window_and_its_claim(self) -> None:
        app = AgenticTuiApp(title="E01 — Agent = Model + Harness", subtitle="The model proposes.")
        assert app.title == "E01 — Agent = Model + Harness"
        assert app.sub_title == "The model proposes."

    def test_an_episode_can_add_its_own_commands(self) -> None:
        app = AgenticTuiApp(commands=[HarnessCommand()])
        assert "harness" in [command.name for command in app._commands.all_commands()]


class TestARebuildKeepsTheConversation:
    """Reported from a live run: `/harness off` asked for the code again.

    Verification, the order menu and the transcript live on the agent
    object, so composing a new one dropped them and the customer had to
    authenticate before every comparison. A rebuild changes which
    controls are wired, not who is on the other end — and `/harness off`
    exists precisely to be asked the *same* question again.
    """

    def _app(self, pool: SqliteConnectionPool) -> AgenticTuiApp:
        return AgenticTuiApp(lambda reporter: build_agent(llm=ScriptedLLM(), pool=pool))

    def test_a_verified_session_survives(
        self, seeded_pool: SqliteConnectionPool, restore_settings: None
    ) -> None:
        app = self._app(seeded_pool)
        app._agent._sessions["s1"] = Customer(customer_id=1, name="Ada", email="ada@example.com")
        app._apply(ConsoleEffect(rebuild_agent=True))
        assert "s1" in app._agent._sessions

    def test_the_transcript_survives(
        self, seeded_pool: SqliteConnectionPool, restore_settings: None
    ) -> None:
        app = self._app(seeded_pool)
        app._agent._conversations["s1"] = [Message(role="user", content="where is order 10001")]
        app._apply(ConsoleEffect(rebuild_agent=True))
        assert app._agent._conversations["s1"]

    def test_it_is_still_a_different_agent(
        self, seeded_pool: SqliteConnectionPool, restore_settings: None
    ) -> None:
        """Carrying the conversation must not quietly skip the rebuild."""
        app = self._app(seeded_pool)
        before = app._agent
        app._apply(ConsoleEffect(rebuild_agent=True))
        assert app._agent is not before

    def test_an_agent_with_no_conversation_to_carry_still_rebuilds(self) -> None:
        """The console composes whatever it was given, not only this agent."""
        app = AgenticTuiApp(lambda reporter: SimpleNamespace(memory=None, handle=None))
        before = app._agent
        app._apply(ConsoleEffect(rebuild_agent=True))
        assert app._agent is not before
