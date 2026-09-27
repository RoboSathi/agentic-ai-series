"""Console slash commands and how they draw (LLD §20).

Two things are under test here. The commands themselves — do they return
the right data, and do they fail usefully when they cannot — and the
dispatch grammar, which is the only place a typed line becomes a command.

Every command that reads the database is given the in-memory fixture
pool. A command that opened the configured database would pass or fail
depending on what the developer's dummy data happened to contain that
afternoon, which is not a test.
"""

import pytest

from agentic_commons.commands import (
    CommandError,
    CommandRegistry,
    CommandResult,
    CommandRow,
)
from agentic_commons.commands.harness import HarnessCommand
from agentic_commons.commands.inspection import AuditCommand, ConfigCommand
from agentic_commons.commands.system import ClearCommand
from agentic_commons.config import active_features, get_settings, set_active_features
from agentic_commons.persistence import SqliteConnectionPool
from agentic_commons.tui.command_view import CommandView
from agentic_commons.tui.suggest import CommandSuggester
from customer_support_agent.console.commands import QueueCommand


class TestDispatch:
    def test_a_slash_line_is_a_command_and_a_sentence_is_not(self) -> None:
        assert CommandRegistry.is_command("/audit")
        assert not CommandRegistry.is_command("where is my order")

    @pytest.mark.asyncio
    async def test_an_unknown_command_suggests_near_matches(self) -> None:
        # `/queue` is the agent's, not the platform's, so it is registered
        # the way the console registers it: from the running host.
        registry = CommandRegistry()
        registry.register(QueueCommand())
        result = await registry.dispatch("/que")
        assert not result.ok
        assert "/queue" in result.lines[0]

    @pytest.mark.asyncio
    async def test_a_bare_slash_shows_help(self) -> None:
        result = await CommandRegistry().dispatch("/")
        assert result.title == "Commands"

    @pytest.mark.asyncio
    async def test_a_failing_command_reports_its_usage_rather_than_raising(self) -> None:
        """A `CommandError` is an answer, not a crash — the console stays up."""
        result = await CommandRegistry().dispatch("/eval case")
        assert not result.ok
        assert "/eval" in result.lines[-1]

    def test_completions_narrow_as_you_type(self) -> None:
        registry = CommandRegistry()
        assert registry.completions("/c") == ["clear", "config"]
        assert registry.completions("/m") == ["memory"]

    @pytest.mark.asyncio
    async def test_the_suggester_completes_only_the_command_word(self) -> None:
        """Arguments are values — guessing one would be guessing intent."""
        suggester = CommandSuggester(CommandRegistry())
        assert await suggester.get_suggestion("/au") == "/audit"
        assert await suggester.get_suggestion("/audit den") is None
        assert await suggester.get_suggestion("where is my order") is None


class TestInspectionCommands:
    @pytest.mark.asyncio
    async def test_config_lists_sections_then_drills_into_one(self) -> None:
        sections = await ConfigCommand().run([])
        assert any(row.cells[0] == "gatekeeper" for row in sections.rows)
        detail = await ConfigCommand().run(["gatekeeper"])
        assert detail.rows

    @pytest.mark.asyncio
    async def test_config_never_renders_a_secret(self) -> None:
        """API keys live in bare env vars, so they cannot appear here.

        This asserts the property rather than the implementation: if a
        key were ever added to the YAML, `/config` would print it to
        whoever is watching the screen.
        """
        for section in (await ConfigCommand().run([])).rows:
            detail = await ConfigCommand().run([section.cells[0]])
            rendered = " ".join(cell for row in detail.rows for cell in row.cells).casefold()
            assert "api_key" not in rendered
            assert "nvapi-" not in rendered

    @pytest.mark.asyncio
    async def test_an_unknown_config_section_is_an_error_not_an_empty_table(self) -> None:
        registry = CommandRegistry(commands=[ConfigCommand()])
        result = await registry.dispatch("/config nosuchsection")
        assert not result.ok

    @pytest.mark.asyncio
    async def test_an_empty_audit_trail_is_a_result_not_a_failure(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        result = await AuditCommand(seeded_pool).run([])
        assert result.ok
        assert result.lines

    @pytest.mark.asyncio
    async def test_the_queue_reports_nothing_pending_without_erroring(
        self, seeded_pool: SqliteConnectionPool
    ) -> None:
        result = await QueueCommand(seeded_pool).run([])
        assert result.ok
        assert "0 pending" in result.title


class TestSystemCommands:
    @pytest.mark.asyncio
    async def test_help_lists_every_registered_command(self) -> None:
        registry = CommandRegistry()
        result = await registry.dispatch("/help")
        assert len(result.rows) == len(registry.all_commands())

    @pytest.mark.asyncio
    async def test_help_explains_one_command(self) -> None:
        registry = CommandRegistry()
        registry.register(QueueCommand())
        result = await registry.dispatch("/help queue")
        assert result.title == "/queue"
        assert result.lines[-1].startswith("/queue")

    @pytest.mark.asyncio
    async def test_clear_asks_the_console_rather_than_touching_a_widget(self) -> None:
        """A command has no handle on the UI, so it states the effect."""
        result = await ClearCommand().run([])
        assert result.effect.clear_transcript
        assert result.empty


class TestCommandView:
    def test_columns_align_to_their_widest_cell(self) -> None:
        result = CommandResult(title="T", headers=["id", "name"])
        result.row("1", "Ada").row("1000", "Grace")
        lines = [line.plain for line in CommandView().render(result)]
        assert lines[1:] == ["id    name ", "1     Ada  ", "1000  Grace"]

    def test_a_long_cell_is_elided_rather_than_pushing_columns_off_screen(self) -> None:
        width = CommandView()._style.max_column_width
        result = CommandResult(headers=["v"])
        result.row("x" * (width + 20))
        rendered = CommandView().render(result)[-1].plain
        assert len(rendered) == width
        assert rendered.endswith("…")

    def test_a_failed_result_is_styled_as_an_error(self) -> None:
        view = CommandView()
        lines = view.render(CommandResult(title="Unknown command", ok=False).line("nope"))
        assert all(line.style == view._style.error_style for line in lines)

    def test_a_row_keeps_its_own_style(self) -> None:
        """Pending review is yellow wherever it appears — the row says so."""
        result = CommandResult(rows=[CommandRow(cells=["10001"], style="yellow")])
        assert CommandView().render(result)[0].style == "yellow"

    def test_the_rendering_copies_out_as_plain_text(self) -> None:
        """Ctrl-O exports the feed, so a table must survive being pasted."""
        result = CommandResult(title="Orders", headers=["order"])
        result.row("10001")
        text = "\n".join(line.plain for line in CommandView().render(result))
        assert "10001" in text
        assert "│" not in text


class TestHarnessCommand:
    """E01's switch: the same model, with and without what surrounds it.

    The harness *is* the feature tree, so these assert against it rather
    than a separate list that could drift from what the builder reads.
    """

    def _features(self) -> object:
        return active_features()

    async def test_it_reports_every_feature(self, restore_settings: None) -> None:
        set_active_features(get_settings().features)
        result = await HarnessCommand().run([])
        assert len(result.rows) == len(active_features().paths())

    async def test_off_withholds_every_feature(self, restore_settings: None) -> None:
        set_active_features(get_settings().features)
        await HarnessCommand().run(["off"])
        features = active_features()
        assert not any(features.is_on(path) for path in features.paths())

    async def test_on_grants_every_feature(self, restore_settings: None) -> None:
        set_active_features(get_settings().features)
        await HarnessCommand().run(["off"])
        await HarnessCommand().run(["on"])
        features = active_features()
        assert all(features.is_on(path) for path in features.paths())

    async def test_a_named_feature_moves_alone(self, restore_settings: None) -> None:
        set_active_features(get_settings().features)
        await HarnessCommand().run(["hallucination_guard", "off"])
        features = active_features()
        assert features.is_on("hallucination_guard") is False
        assert features.is_on("gatekeeper") is True

    async def test_a_sub_feature_can_be_named(self, restore_settings: None) -> None:
        set_active_features(get_settings().features)
        await HarnessCommand().run(["memory.episodic", "off"])
        features = active_features()
        assert features.is_on("memory.episodic") is False
        assert features.is_on("memory.working") is True

    async def test_switching_a_parent_off_takes_its_children(
        self, restore_settings: None
    ) -> None:
        set_active_features(get_settings().features)
        await HarnessCommand().run(["memory", "off"])
        assert active_features().is_on("memory.semantic") is False

    async def test_it_never_reads_from_disk(self, restore_settings: None) -> None:
        """A hot reload. `on` grants everything rather than restoring a file,
        so the console never pauses on I/O mid-demonstration."""
        set_active_features(get_settings().features.with_override("memory", False))
        await HarnessCommand().run(["on"])
        assert active_features().is_on("memory") is True

    async def test_every_mutating_form_asks_for_a_rebuild(
        self, restore_settings: None
    ) -> None:
        """The composition root decides what to wire, so a running agent
        was built under the previous answer."""
        set_active_features(get_settings().features)
        for args in (["off"], ["on"], ["gatekeeper", "off"]):
            assert (await HarnessCommand().run(args)).effect.rebuild_agent

    async def test_reporting_alone_rebuilds_nothing(self, restore_settings: None) -> None:
        set_active_features(get_settings().features)
        assert not (await HarnessCommand().run([])).effect.rebuild_agent

    async def test_it_rejects_anything_but_on_or_off(self) -> None:
        with pytest.raises(CommandError):
            await HarnessCommand().run(["maybe"])

    async def test_an_unknown_feature_names_the_real_ones(
        self, restore_settings: None
    ) -> None:
        """A typo should cost a glance, not a silent no-op."""
        set_active_features(get_settings().features)
        with pytest.raises(CommandError, match="hallucination_guard"):
            await HarnessCommand().run(["hallucinaton_guard", "off"])

    def test_auth_is_not_a_feature(self, restore_settings: None) -> None:
        """Identity is not an approval control.

        Switching RBAC off does not make the agent reckless, it makes it
        unable to authenticate anybody — so it can do nothing at all,
        which is the opposite of the demonstration E01 is making.
        """
        assert "security.auth" not in get_settings().features.paths()
        assert "auth" not in get_settings().features.paths()


class TestConfigStaysReadOnly:
    """`/config` shows; `/harness` changes. One command per verb.

    Worth asserting rather than assuming: an operator who can change
    configuration from the inspection command has two ways to move the
    same state, and only one of them rebuilds the agent afterwards.
    """

    @pytest.mark.asyncio
    async def test_the_feature_tree_is_visible(self) -> None:
        result = await ConfigCommand().run(["features"])
        shown = {row.cells[0] for row in result.rows}
        assert "gatekeeper" in shown
        assert "memory.working" in shown

    @pytest.mark.asyncio
    async def test_showing_it_changes_nothing(self, restore_settings: None) -> None:
        set_active_features(get_settings().features)
        before = active_features()
        await ConfigCommand().run(["features"])
        assert active_features().tree == before.tree

    @pytest.mark.asyncio
    async def test_it_never_asks_for_a_rebuild(self) -> None:
        """Nothing it does could require one."""
        assert not (await ConfigCommand().run(["features"])).effect.rebuild_agent

    @pytest.mark.asyncio
    async def test_it_offers_no_way_to_write(self) -> None:
        registry = CommandRegistry(commands=[ConfigCommand()])
        result = await registry.dispatch("/config features gatekeeper off")
        assert not result.ok


class TestConsoleSurface:
    """The console offers exactly these commands — a removed one cannot creep back."""

    KEPT = {"audit", "clear", "config", "eval", "harness", "help", "kb", "memory", "queue"}

    def test_the_platform_and_the_agent_together_register_the_kept_set(self) -> None:
        from customer_support_agent.console import CustomerSupportHost

        registry = CommandRegistry()
        for command in CustomerSupportHost().commands():
            registry.register(command)
        assert {command.name for command in registry.all_commands()} == self.KEPT

    @pytest.mark.asyncio
    async def test_help_groups_every_command_once(self) -> None:
        result = await CommandRegistry().dispatch("/help")
        names = [row.cells[1] for row in result.rows]
        assert len(names) == len(set(names))
        assert all(row.cells[0] for row in result.rows)
