"""A console note keeps its configured colour, and an action button is one row.

`FeedNote` used to apply its colour as the Text's base style. Textual parses a
base style itself and does not know Rich's colour names, so `orange3` (the
TURN COMPLETE footer) and `bright_black` (muted notes) came out as the default
white. Applied as a span, every Rich colour survives.
"""

from textual.app import App, ComposeResult
from textual.widgets import Button

from agent_platform.interface.host.protocols import HostAction
from agent_platform.interface.tui.app import AgenticTuiApp
from agent_platform.interface.tui.widgets import ActionBar, FeedNote

# Textual's default text colour: what a dropped style falls back to.
_DEFAULT_WHITE = "#e0e0e0"


class _Notes(App[None]):
    def compose(self) -> ComposeResult:
        yield FeedNote("── TURN 01 COMPLETE ──", "orange3")
        yield FeedNote("a muted note", "bright_black")


class TestANoteKeepsItsColour:
    async def test_rich_colour_names_are_not_dropped(self) -> None:
        app = _Notes()
        async with app.run_test(size=(60, 6)):
            for note in app.query(FeedNote):
                [first, *_] = [s for s in note.render_line(0) if s.text.strip()]
                assert first.style is not None and first.style.color is not None
                assert first.style.color.name != _DEFAULT_WHITE


class _Bar(App[None]):
    # The console's own stylesheet, so the button is sized as the console sizes it.
    CSS = AgenticTuiApp.CSS

    def compose(self) -> ComposeResult:
        yield ActionBar()


class TestAnActionButtonIsOneRow:
    async def test_it_is_one_row_tall(self) -> None:
        app = _Bar()
        async with app.run_test(size=(80, 10)) as pilot:
            app.query_one(ActionBar).offer(
                [HostAction(label="■ Stop Agent", command="cancel", variant="error")], busy=False
            )
            await pilot.pause()
            assert app.query_one(Button).size.height == 1
