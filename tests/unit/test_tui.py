"""The live status feed's labels, widgets and clipboard (LLD §20)."""

from pathlib import Path
from types import SimpleNamespace

from agent_platform.commands import ConsoleEffect
from agent_platform.commands.harness import HarnessCommand
from agent_platform.persistence import SqliteConnectionPool
from agent_platform.tui import (
    Osc52ClipboardService,
    StageLabelRegistry,
    StatusLineWidget,
    TokenCounterWidget,
)
from agent_platform.tui.app import AgenticTuiApp
from agent_platform.tui.widgets import _compact, _elapsed
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

    def test_a_namespaced_tool_resolves_to_the_same_wording(self) -> None:
        registry = StageLabelRegistry()
        # Every spelling reads as the tool's model name (ADR 0010).
        assert registry.tool_label("support.process_refund") == registry.tool_label(
            "support__process_refund"
        )


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

    def test_the_footer_and_status_line_share_one_metrics_line(self) -> None:
        """`207.01s` read as 3m 27s once the status line formatted it; the footer did not."""
        line = StatusLineWidget()
        counter = TokenCounterWidget()
        line.bind_tokens(counter)
        line.begin("Working…")
        line._started_at -= 207
        counter.add(9_700)
        line.finish()
        assert line.metrics() == "3m 27s · 9.7k tokens"
        assert "(3m 27s · 9.7k tokens)" in line.render().plain


class TestLayout:
    """The prompt sits under the transcript, with the live status just above it.

    Reported as "the chat window is too low": `dock: bottom` pins a
    widget to the screen edge and removes it from the flow, which left
    the prompt stranded at the very bottom. Nothing is docked now.
    """

    async def test_the_prompt_is_below_the_transcript(self) -> None:
        from textual.containers import VerticalScroll

        from agent_platform.tui.app import AgenticTuiApp, ChatPrompt

        app = AgenticTuiApp()
        async with app.run_test(size=(78, 26)):
            feed = app.query_one("#feed", VerticalScroll)
            prompt = app.query_one("#query", ChatPrompt)
            assert prompt.region.y > feed.region.y

    async def test_the_status_line_sits_above_the_prompt(self) -> None:
        """Between the transcript and the chat bar, as in Claude Code."""
        from agent_platform.tui.app import AgenticTuiApp, ChatPrompt

        app = AgenticTuiApp()
        async with app.run_test(size=(78, 26)):
            prompt = app.query_one("#query", ChatPrompt)
            feed = app.query_one("#feed")
            assert feed.region.y + feed.region.height <= app._status.region.y
            assert app._status.region.y + app._status.region.height <= prompt.region.y

    async def test_nothing_is_docked(self) -> None:
        """Three docked widgets once landed in the same place."""
        from agent_platform.tui.app import AgenticTuiApp

        assert "dock:" not in AgenticTuiApp.CSS.replace(" ", "")

    async def test_the_transcript_takes_the_remaining_height(self) -> None:
        from textual.containers import VerticalScroll

        from agent_platform.tui.app import AgenticTuiApp

        app = AgenticTuiApp()
        async with app.run_test(size=(78, 26)):
            assert app.query_one("#feed", VerticalScroll).region.height > 10

    async def test_one_session_spans_the_whole_window(self) -> None:
        """Keying per turn would give every message its own conversation."""
        from agent_platform.tui.app import AgenticTuiApp

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


class TestEveryKeptCommandIsWired:
    """Boot the console and run every command it offers, end to end."""

    KEPT = ("/help", "/config", "/audit", "/memory", "/eval", "/kb", "/queue", "/harness", "/clear")

    def _app(self, pool: object) -> AgenticTuiApp:
        from agent_platform.commands.knowledge import KnowledgeBaseCommand
        from agent_platform.memory import WorkingMemoryStore
        from customer_support_agent.review.handler import support_queue_command

        agent = SimpleNamespace(working_memory=WorkingMemoryStore(), handle=None)
        commands = [
            support_queue_command(pool, agent=lambda: agent),
            KnowledgeBaseCommand(lambda: None),
        ]
        return AgenticTuiApp(lambda reporter: agent, commands=commands)

    async def test_each_command_answers_without_crashing(self, seeded_pool: object) -> None:
        app = self._app(seeded_pool)
        async with app.run_test(size=(100, 40)):
            for line in self.KEPT:
                result = await app._commands.dispatch(line)
                # /kb has no knowledge base in this app: an answer, not a crash.
                assert result.ok or line == "/kb", (line, result.lines)

    async def test_typing_a_command_renders_it_in_the_feed(self, seeded_pool: object) -> None:
        from agent_platform.tui.app import ChatPrompt

        app = self._app(seeded_pool)
        async with app.run_test(size=(100, 40)) as pilot:
            app.query_one("#query", ChatPrompt).text = "/help"
            await pilot.press("enter")
            await pilot.pause()
            assert "/queue" in app._panel.as_text()

    async def test_memory_is_rebound_after_a_rebuild(self, seeded_pool: object) -> None:
        app = self._app(seeded_pool)
        async with app.run_test(size=(100, 40)):
            await app._run_command("/harness off")
            result = await app._commands.dispatch("/memory working")
            assert result.ok
            await app._run_command("/harness on")


class TestTheMultiLinePromptCompletes:
    """The prompt became a TextArea and Tab still read `Input.value`: `/har` + Tab crashed."""

    async def test_tab_completes_a_command_word(self) -> None:
        from agent_platform.tui.app import AgenticTuiApp, ChatPrompt

        app = AgenticTuiApp()
        async with app.run_test(size=(100, 30)) as pilot:
            prompt = app.query_one("#query", ChatPrompt)
            prompt.text = "/har"
            await pilot.press("tab")
            assert prompt.text == "/harness "
            assert prompt.cursor_location == prompt.document.end

    async def test_a_second_tab_moves_on_to_the_argument(self) -> None:
        from agent_platform.tui.app import AgenticTuiApp, ChatPrompt

        app = AgenticTuiApp()
        async with app.run_test(size=(100, 30)) as pilot:
            prompt = app.query_one("#query", ChatPrompt)
            prompt.text = "/har"
            await pilot.press("tab", "tab")
            assert prompt.text.startswith("/harness ") and len(prompt.text) > len("/harness ")

    async def test_typing_dismisses_an_offered_menu(self) -> None:
        from types import SimpleNamespace

        from agent_platform.tui.app import AgenticTuiApp

        app = AgenticTuiApp()
        async with app.run_test(size=(100, 30)) as pilot:
            app._menu.offer([SimpleNamespace(label="Order 10533", value="10533")])
            assert app._menu.offered
            await pilot.press("x")
            await pilot.pause()
            assert not app._menu.offered

    async def test_shift_enter_breaks_the_line_and_enter_does_not(self) -> None:
        from agent_platform.tui.app import AgenticTuiApp, ChatPrompt

        app = AgenticTuiApp()
        async with app.run_test(size=(100, 30)) as pilot:
            prompt = app.query_one("#query", ChatPrompt)
            await pilot.press("a", "shift+enter", "b")
            assert prompt.text == "a\nb"

    async def test_a_commands_output_is_followed_to_its_end(self) -> None:
        """The feed used to land on the last line of a command's output, and stopped short."""
        from agent_platform.tui.app import AgenticTuiApp, ChatPrompt

        app = AgenticTuiApp()
        # Short enough that /help's output cannot fit, whatever the panel's chrome.
        async with app.run_test(size=(100, 12)) as pilot:
            app.query_one("#query", ChatPrompt).text = "/help"
            await pilot.press("enter")
            await pilot.pause()
            await pilot.pause()
            panel = app._panel
            assert panel.max_scroll_y > 0, "the output must overflow for this to test anything"
            assert panel.scroll_y >= panel.max_scroll_y - 1, (panel.scroll_y, panel.max_scroll_y)
