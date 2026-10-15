"""Console slash commands and how they draw (LLD §20).

Two things are under test here. The commands themselves — do they return
the right data, and do they fail usefully when they cannot — and the
dispatch grammar, which is the only place a typed line becomes a command.

Every command that reads the database is given the in-memory fixture
pool. A command that opened the configured database would pass or fail
depending on what the developer's dummy data happened to contain that
afternoon, which is not a test.
"""

from typing import Any

import pytest

from agent_platform.audit import DuckDbAuditLedger
from agent_platform.config import active_features, get_settings, set_active_features
from agent_platform.core.persistence import SqliteConnectionPool
from agent_platform.interface.commands import (
    CommandError,
    CommandRegistry,
    CommandResult,
    CommandRow,
)
from agent_platform.interface.commands.harness import HarnessCommand
from agent_platform.interface.commands.inspection import AuditCommand
from agent_platform.interface.commands.review import QueueCommand
from agent_platform.interface.commands.system import ClearCommand
from agent_platform.interface.presentation.command_view import CommandView
from agent_platform.observability.store import DuckDbObservabilityStore

# A key value that must never be rendered, whatever variable holds it.
SECRET = "sk-test-never-shown"


class TestDispatch:
    def test_a_slash_line_is_a_command_and_a_sentence_is_not(self) -> None:
        assert CommandRegistry.is_command("/audit")
        assert not CommandRegistry.is_command("where is my order")

    @pytest.mark.asyncio
    async def test_an_unknown_command_suggests_near_matches(self) -> None:
        # `/review` is the platform's one review queue (ADR P-14).
        registry = CommandRegistry()
        result = await registry.dispatch("/rev")
        assert not result.ok
        assert "/review" in result.lines[0]

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
        assert registry.completions("/c") == ["clear"]
        assert registry.completions("/m") == ["memory"]


class TestInspectionCommands:
    @pytest.mark.asyncio
    async def test_an_empty_audit_trail_is_a_result_not_a_failure(self) -> None:
        ledger = DuckDbAuditLedger(DuckDbObservabilityStore(":memory:"))
        result = await AuditCommand(ledger).run([])
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
        result = await registry.dispatch("/help review")
        assert result.title == "/review"
        assert result.lines[-1].startswith("/review")

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
        # Styles are spans (Textual drops Rich colour names from a base style).
        assert all(_style_of(line) == view._style.error_style for line in lines)

    def test_a_row_keeps_its_own_style(self) -> None:
        """Pending review is yellow wherever it appears — the row says so."""
        result = CommandResult(rows=[CommandRow(cells=["10001"], style="yellow")])
        assert _style_of(CommandView().render(result)[0]) == "yellow"

    def test_the_rendering_copies_out_as_plain_text(self) -> None:
        """Ctrl-O exports the feed, so a table must survive being pasted."""
        result = CommandResult(title="Orders", headers=["order"])
        result.row("10001")
        text = "\n".join(line.plain for line in CommandView().render(result))
        assert "10001" in text
        assert "│" not in text


STORES = ("working", "episodic", "semantic", "procedural")


class TestHarnessCommand:
    """The harness switch: the same model, with and without what surrounds it.

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

    async def test_switching_a_parent_off_takes_its_children(self, restore_settings: None) -> None:
        set_active_features(get_settings().features)
        await HarnessCommand().run(["memory", "off"])
        assert active_features().is_on("memory.semantic") is False

    async def test_switching_a_parent_on_restores_its_children(
        self, restore_settings: None
    ) -> None:
        set_active_features(get_settings().features)
        await HarnessCommand().run(["off"])
        await HarnessCommand().run(["memory", "on"])
        features = active_features()
        assert all(features.is_on(f"memory.{store}") for store in STORES)

    async def test_a_sub_feature_can_be_named_in_words(self, restore_settings: None) -> None:
        """`/harness memory working off` is the same as `memory.working off`."""
        set_active_features(get_settings().features)
        await HarnessCommand().run(["memory", "working", "off"])
        features = active_features()
        assert features.is_on("memory.working") is False
        assert features.is_on("memory.episodic") is True

    def test_after_a_parent_its_children_are_offered(self, restore_settings: None) -> None:
        set_active_features(get_settings().features)
        offered = HarnessCommand().argument_completions(["memory"])
        assert set(STORES) <= set(offered) and {"on", "off"} <= set(offered)

    def test_after_a_leaf_only_the_switch_is_offered(self, restore_settings: None) -> None:
        set_active_features(get_settings().features)
        assert HarnessCommand().argument_completions(["pii"]) == ["on", "off"]

    async def test_it_never_reads_from_disk(self, restore_settings: None) -> None:
        """A hot reload. `on` grants everything rather than restoring a file,
        so the console never pauses on I/O mid-demonstration."""
        set_active_features(get_settings().features.with_override("memory", False))
        await HarnessCommand().run(["on"])
        assert active_features().is_on("memory") is True

    async def test_every_mutating_form_asks_for_a_rebuild(self, restore_settings: None) -> None:
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

    async def test_an_unknown_feature_names_the_real_ones(self, restore_settings: None) -> None:
        """A typo should cost a glance, not a silent no-op."""
        set_active_features(get_settings().features)
        with pytest.raises(CommandError, match="hallucination_guard"):
            await HarnessCommand().run(["hallucinaton_guard", "off"])

    def test_auth_is_not_a_feature(self, restore_settings: None) -> None:
        """Identity is not an approval control.

        Switching RBAC off does not make the agent reckless, it makes it
        unable to authenticate anybody — so it can do nothing at all,
        which is the opposite of what switching the harness off demonstrates.
        """
        assert "security.auth" not in get_settings().features.paths()
        assert "auth" not in get_settings().features.paths()


class TestConsoleSurface:
    """The console offers exactly these commands — a removed one cannot creep back."""

    REMOVED = {"config", "dlq", "queue", "settle"}

    def test_removed_commands_stay_removed(self) -> None:
        names = {command.name for command in CommandRegistry().all_commands()}
        assert not names & self.REMOVED

    def test_the_support_console_has_no_settle(self) -> None:
        from customer_support_agent.console import CustomerSupportHost

        names = {command.name for command in CustomerSupportHost().commands()}
        assert "settle" not in names and "void" in names

    @pytest.mark.asyncio
    async def test_help_groups_every_command_once(self) -> None:
        result = await CommandRegistry().dispatch("/help")
        names = [row.cells[1] for row in result.rows]
        assert len(names) == len(set(names))
        assert all(row.cells[0] for row in result.rows)


def _style_of(line: Any) -> str:
    """The one style a rendered command line is drawn in."""
    return str(line.spans[0].style) if line.spans else str(line.style)
