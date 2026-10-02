"""The buttons beside the prompt survive being re-offered; the answer is coloured by config."""

from dataclasses import dataclass

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button

from agentic_commons.config import get_settings
from agentic_commons.config.settings import ResponseHighlight
from agentic_commons.tui.widgets import ActionBar, render_response


@dataclass(frozen=True)
class _Action:
    label: str
    command: str
    during_turn: bool = False


class _Bar(App[None]):
    def compose(self) -> ComposeResult:
        yield ActionBar()


@pytest.mark.asyncio
async def test_re_offering_actions_replaces_the_buttons_instead_of_failing() -> None:
    """Reusing `action-0` while the old button was still being removed raised, and the bar
    stayed empty for the rest of the session."""
    app = _Bar()
    async with app.run_test() as pilot:
        bar = app.query_one(ActionBar)
        bar.offer([_Action("Done interviewing", "/done")], busy=False)
        await pilot.pause()
        bar.offer([_Action("Done interviewing", "/done")], busy=True)
        bar.offer([_Action("Approve and run", "/approve")], busy=False)
        await pilot.pause()
        buttons = list(bar.query(Button))
        assert [str(b.label) for b in buttons] == ["Approve and run"]
        assert bar.command_for(buttons[0].id) == "/approve"


def test_configured_highlights_colour_an_agents_own_vocabulary() -> None:
    tui = get_settings().tui
    original = tui.response_highlights
    tui.response_highlights = [
        ResponseHighlight(pattern=r"\[must\]", style="bold red"),
        ResponseHighlight(pattern=r"^→ Next:.*$", style="bold green"),
    ]
    try:
        text = render_response("R1 [must] cites.\n\n→ Next: /approve", tui.response_styles)
    finally:
        tui.response_highlights = original
    styled = {text.plain[s.start : s.end]: str(s.style) for s in text.spans}
    assert styled["[must]"] == "bold red"
    assert styled["→ Next: /approve"] == "bold green"


def test_a_highlight_drops_markdown_bold_markers_from_what_it_shows() -> None:
    tui = get_settings().tui
    original = tui.response_highlights
    tui.response_highlights = [
        ResponseHighlight(pattern=r"^\*\*Status:\*\* COMPLETE.*$", style="green")
    ]
    try:
        text = render_response("**Status:** COMPLETE · 3 sources", tui.response_styles)
    finally:
        tui.response_highlights = original
    assert text.plain == "Status: COMPLETE · 3 sources"
