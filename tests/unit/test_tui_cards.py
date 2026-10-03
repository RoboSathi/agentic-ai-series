"""A card reads by its state: the title keeps its outcome colour (even focused), a pending card
is settled in place, and an expanded card shows keys apart from values, values by state."""

from typing import Any

import pytest
from rich.text import Text
from textual.app import App, ComposeResult
from textual.widgets._collapsible import CollapsibleTitle

from agent_platform.config import get_settings
from agent_platform.observability.types import StatusEvent
from agent_platform.tui.app import AgenticTuiApp
from agent_platform.tui.widgets import (
    LiveStatusPanel,
    StageEventWidget,
    StatusLineWidget,
    _append_line,
    _append_pair,
)


class _Feed(App[None]):
    CSS = AgenticTuiApp.CSS

    def compose(self) -> ComposeResult:
        yield LiveStatusPanel()


def _event(stage: str, status: Any, **extra: Any) -> StatusEvent:
    return StatusEvent(
        run_id="r",
        stage=stage,
        status=status,
        event_kind=extra.pop("event_kind", "narrative"),
        summary=extra.pop("summary", "hello"),
        fields=extra.pop("fields", {"Section": ["line"]}),
    )


def _title_colour(widget: StageEventWidget) -> str:
    """The colour the title's text is drawn in (its label, after the symbol)."""
    segments = [s for s in widget.query_one(CollapsibleTitle).render_line(0) if s.text.strip()]
    style = segments[-1].style
    assert style is not None and style.color is not None
    return f"#{style.color.triplet.hex[1:]}".upper() if style.color.triplet else style.color.name


def _expected(status: str) -> str:
    return get_settings().tui.outcome_colors[status].upper()


@pytest.mark.asyncio
async def test_a_focused_title_keeps_its_outcome_colour() -> None:
    app = _Feed()
    async with app.run_test(size=(100, 30)) as pilot:
        panel = app.query_one(LiveStatusPanel)
        panel.show_status(_event("thought", "completed"))
        panel.show_status(_event("budget_tick", "warning"))
        await pilot.pause()
        for widget in app.query(StageEventWidget):
            await pilot.click(widget.query_one(CollapsibleTitle))
            await pilot.pause()
            assert _title_colour(widget) == _expected(widget.event.status)


@pytest.mark.asyncio
async def test_a_pending_card_is_settled_in_place() -> None:
    app = _Feed()
    async with app.run_test(size=(100, 30)) as pilot:
        panel = app.query_one(LiveStatusPanel)
        panel.show_status(_event("thought", "started", event_kind="model"))
        panel.show_status(_event("note", "completed"))
        await pilot.pause()
        panel.show_status(_event("thought", "completed", event_kind="model"))
        await pilot.pause()
        cards = list(app.query(StageEventWidget))
        assert [c.event.stage for c in cards] == ["thought", "note"]
        assert cards[0].event.status == "completed"


@pytest.mark.asyncio
async def test_a_started_event_of_another_kind_draws_no_card() -> None:
    app = _Feed()
    async with app.run_test(size=(100, 30)) as pilot:
        app.query_one(LiveStatusPanel).show_status(_event("search", "started", event_kind="stage"))
        await pilot.pause()
        assert not list(app.query(StageEventWidget))


def _spans(text: Text) -> dict[str, str]:
    return {text.plain[s.start : s.end]: str(s.style) for s in text.spans}


def test_a_key_and_its_value_take_different_colours() -> None:
    styles = get_settings().tui.body_styles
    out = Text()
    _append_pair(out, "matches", " = ", "3")
    spans = _spans(out)
    assert spans["matches"] == styles["key"] and spans["3"] == styles["value"]


def test_a_value_naming_a_state_takes_that_states_tone() -> None:
    tones = get_settings().tui.tone_styles
    out = Text()
    _append_pair(out, "deterministic", " = ", "MATCH")
    assert _spans(out)["MATCH"] == tones["fail"]


def test_a_written_key_value_line_is_split_and_its_value_toned() -> None:
    styles, tones = get_settings().tui.body_styles, get_settings().tui.tone_styles
    out = Text()
    _append_line(out, "Refusal rules: no match", tones["pass"])
    spans = _spans(out)
    assert spans["Refusal rules"] == styles["key"] and spans["no match"] == tones["pass"]


def test_a_url_is_not_mistaken_for_a_key() -> None:
    out = Text()
    _append_line(out, "https://example.com/page", "")
    assert len(out.spans) == 1 and out.plain == "https://example.com/page"


def test_the_status_block_can_expand_with_nothing_always_shown() -> None:
    line = StatusLineWidget()
    line.begin("Researching")
    line.set_block([], ["RUNNING", "Scope coverage 7/7"])
    assert "RUNNING" not in line.render().plain
    line.toggle_detail()
    assert "RUNNING" in line.render().plain
