"""The live status feed's widgets (LLD §20).

`LiveStatusPanel` is an Observer on `LiveStatusFeed` — stages push, the
panel never polls. That direction matters: a polling UI either lags the
work or burns CPU asking whether anything happened.

Everything rendered here is a summary. The full structured record of the
same event is already in `logs/`, so the feed can stay readable without
anyone losing the detail.
"""

import time

from rich.text import Text
from textual.reactive import reactive
from textual.widgets import Static

from agentic_commons.config import get_settings
from agentic_commons.observability.types import StatusEvent
from agentic_commons.tui.labels import StageLabelRegistry

_MARKS = {"completed": ("✓", "green"), "failed": ("✗", "red")}


class TokenCounterWidget(Static):
    """Session total and current-turn count, both visible at once.

    Both, because they answer different questions: the turn count says
    whether *this* request is running away, the session total says what
    the conversation has cost so far.
    """

    session_tokens: reactive[int] = reactive(0)
    turn_tokens: reactive[int] = reactive(0)

    def __init__(self) -> None:
        super().__init__(id="tokens")

    def add(self, tokens: int) -> None:
        """Account for tokens spent in the current turn."""
        self.turn_tokens += tokens
        self.session_tokens += tokens
        self.refresh()

    def start_turn(self) -> None:
        """Reset the per-turn counter; the session total keeps accruing."""
        self.turn_tokens = 0
        self.refresh()

    def render(self) -> Text:
        """`↓ 1.2k turn · 8.4k session`."""
        if not get_settings().tui.show_token_count:
            return Text("")
        return Text(
            f"↓ {_compact(self.turn_tokens)} turn · {_compact(self.session_tokens)} session",
            style="dim",
        )


class StatusLineWidget(Static):
    """The spinner, elapsed timer and current stage — Claude Code's status line."""

    frame: reactive[int] = reactive(0)
    stage_label: reactive[str] = reactive("")
    running: reactive[bool] = reactive(False)

    def __init__(self) -> None:
        super().__init__(id="status-line")
        settings = get_settings().tui
        self._frames = settings.spinner_frames
        self._started_at = 0.0

    def begin(self, label: str) -> None:
        """Start the timer and show a stage."""
        self._started_at = time.monotonic()
        self.stage_label = label
        self.running = True

    def update_stage(self, label: str) -> None:
        """Change the stage without restarting the timer."""
        self.stage_label = label

    def finish(self) -> None:
        """Stop the spinner; the elapsed time stays on screen."""
        self.running = False
        self.refresh()

    def tick(self) -> None:
        """Advance the spinner one frame."""
        if self.running:
            self.frame = (self.frame + 1) % len(self._frames)
            self.refresh()

    def render(self) -> Text:
        """`⠹ Assembling context… (1m 04s)`."""
        if not self.stage_label:
            return Text("")
        elapsed = _elapsed(time.monotonic() - self._started_at)
        spinner = self._frames[self.frame % len(self._frames)] if self.running else "•"
        line = Text(f"{spinner} ", style="cyan")
        line.append(self.stage_label, style="bold" if self.running else "dim")
        line.append(f" ({elapsed})", style="dim")
        return line


class StatusFeedBuffer:
    """The feed's lines, independent of any widget.

    Separated from the panel so the feed's behaviour — what a failed
    stage looks like, where the backlog is trimmed — is testable without
    a running Textual app. The widget renders this; it does not own it.
    """

    def __init__(self, labels: StageLabelRegistry | None = None) -> None:
        self._labels = labels or StageLabelRegistry()
        self._lines: list[Text] = []
        self._max_lines = get_settings().tui.max_feed_lines

    def show_status(self, event: StatusEvent) -> None:
        """Turn one status event into a feed line.

        `started` events add nothing: they drive the status line instead,
        and a line per start would double every stage in the feed.

        The mark says what happened; the colour says what *kind* of thing
        it was. Both carry information, which is why the feed is not one
        colour with a tick in front of every line.
        """
        if event.status == "started":
            return
        mark, mark_style = _MARKS.get(event.status, ("·", "dim"))
        style = "red" if event.status == "failed" else self._labels.style_for(event.stage)
        line = Text(f"{mark} ", style=mark_style)
        line.append(self._labels.label_for(event.stage, tool_name=event.detail or ""), style=style)
        if event.detail and event.status == "failed":
            line.append(f" — {event.detail}", style="red")
        if event.tokens_delta:
            line.append(f"  ↓{_compact(event.tokens_delta)}", style="bright_black")
        self.append(line)

    def note(self, text: str, style: str = "") -> None:
        """Add a line that did not come from a status event."""
        self.append(Text(text, style=style))

    def append(self, line: Text) -> None:
        """Add a line, trimming the backlog to the configured cap."""
        self._lines.append(line)
        if len(self._lines) > self._max_lines:
            self._lines = self._lines[-self._max_lines :]

    def clear(self) -> None:
        """Drop every line."""
        self._lines.clear()

    def renderable(self) -> Text:
        """The buffer as one renderable block."""
        return Text("\n").join(self._lines)

    def as_text(self) -> str:
        """The feed as plain text — what the copy action exports."""
        return "\n".join(line.plain for line in self._lines)


class LiveStatusPanel(Static):
    """Renders a `StatusFeedBuffer` (Observer on `LiveStatusFeed`)."""

    def __init__(self, labels: StageLabelRegistry | None = None) -> None:
        super().__init__()
        self.buffer = StatusFeedBuffer(labels)

    def show_status(self, event: StatusEvent) -> None:
        """Add a status event to the feed and repaint."""
        self.buffer.show_status(event)
        self._repaint()

    def note(self, text: str, style: str = "") -> None:
        """Add a free-text line and repaint."""
        self.buffer.note(text, style)
        self._repaint()

    def clear_feed(self) -> None:
        """Empty the panel."""
        self.buffer.clear()
        self._repaint()

    def as_text(self) -> str:
        """The visible feed as plain text."""
        return self.buffer.as_text()

    def _repaint(self) -> None:
        """Push the buffer to the widget, tolerating no mounted app.

        The buffer is the source of truth, so a panel that is not yet
        mounted simply has nothing to paint onto — not an error.
        """
        try:
            self.update(self.buffer.renderable())
        except Exception:
            pass


def _compact(tokens: int) -> str:
    """`58.8k` rather than `58800` — the feed is read at a glance."""
    return f"{tokens / 1000:.1f}k" if tokens >= 1000 else str(tokens)


def _elapsed(seconds: float) -> str:
    """`04s`, `1m 04s`, `1h 02m`."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60):02d}m"
