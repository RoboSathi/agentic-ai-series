"""The terminal app (LLD §20).

One shell, one panel this phase: the live status feed. The DB browser,
memory inspector and audit view are designed at interface level and
deliberately not built yet — mounting four panels where three are empty
would make the boundary look settled when it is not.

The app is a *consumer* of `observability`, never a second source of
truth. It subscribes to the same `StatusEvent` stream the harness emits
anyway, and the full structured record of every line shown here is
already in `logs/` whether the TUI is running or not.

Layout note, learned the hard way: Textual's `dock` pins a widget rather
than stacking it, so docking the footer, status line and input to the
bottom puts all three in the same place. They flow instead.
"""

from typing import Any, Callable

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Footer, Header, Input

from agentic_commons.config import get_settings
from agentic_commons.logging import get_logger
from agentic_commons.observability import InMemoryLiveStatusFeed, StageReporter
from agentic_commons.observability.types import StatusEvent
from agentic_commons.tui.clipboard import Osc52ClipboardService
from agentic_commons.tui.labels import StageLabelRegistry
from agentic_commons.tui.widgets import LiveStatusPanel, StatusLineWidget, TokenCounterWidget

# Builds the agent, given the reporter it should emit status through. A
# factory rather than a built agent: the app owns the feed, so the agent
# cannot be constructed until the app exists.
AgentFactory = Callable[[StageReporter], Any]


class AgenticTuiApp(App[None]):
    """Live status feed over a running agent."""

    CSS = """
    Screen { layout: vertical; }
    #feed { height: 1fr; padding: 0 1; }
    #status-line { height: auto; padding: 0 1; }
    #tokens { height: auto; padding: 0 1; }
    #query { dock: bottom; }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+o", "copy_feed", "Copy feed"),
        Binding("ctrl+l", "clear_feed", "Clear"),
    ]

    def __init__(self, agent_factory: "AgentFactory | None" = None) -> None:
        super().__init__()
        self._feed = InMemoryLiveStatusFeed()
        self._agent = agent_factory(StageReporter(self._feed, "tui")) if agent_factory else None
        self._labels = StageLabelRegistry()
        self._copier = Osc52ClipboardService()
        self._panel = LiveStatusPanel(self._labels)
        self._status = StatusLineWidget()
        self._tokens = TokenCounterWidget()
        self._turn = 0
        self._journal = get_logger("tui")

    def compose(self) -> ComposeResult:
        """The single-panel layout."""
        yield Header()
        yield VerticalScroll(self._panel, id="feed")
        yield self._status
        yield self._tokens
        yield Input(placeholder="Ask the support agent…", id="query")
        yield Footer()

    def on_mount(self) -> None:
        """Subscribe to the feed and start the spinner clock."""
        self._feed.subscribe(self._on_status)
        self.set_interval(get_settings().tui.refresh_interval_ms / 1000, self._status.tick)
        self.query_one("#query", Input).focus()
        self._panel.note("Ready. Type a customer query below.", style="dim")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Run one query through the agent."""
        query = event.value.strip()
        if not query:
            return
        event.input.value = ""
        await self._run(query)

    async def _run(self, query: str) -> None:
        """Execute a query, keeping the feed live throughout."""
        self._turn += 1
        self._tokens.start_turn()
        self._panel.note(f"\n> {query}", style="bold")
        self._status.begin(self._labels.label_for("intake"))
        if self._agent is None:
            self._panel.note("No agent wired; feed demo only.", style="yellow")
            self._status.finish()
            return
        try:
            state = await self._agent.handle(query, session_id=f"tui-{self._turn}")
            self._panel.note(f"\n{state.response}", style="bright_green")
        except Exception as exc:
            self._journal.exception("Agent run failed")
            self._panel.note(f"\nFailed: {exc}", style="red")
        finally:
            self._status.finish()

    def _on_status(self, event: StatusEvent) -> None:
        """Feed subscriber: update the panel, status line and counters."""
        self._panel.show_status(event)
        if event.status == "started":
            self._status.update_stage(
                self._labels.label_for(event.stage, tool_name=event.detail or "")
            )
        if event.tokens_delta:
            self._tokens.add(event.tokens_delta)

    def reporter(self, run_id: str) -> StageReporter:
        """A reporter the agent can emit through."""
        return StageReporter(self._feed, run_id)

    def action_copy_feed(self) -> None:
        """Export the visible feed and try the terminal clipboard."""
        path = self._copier.copy(self._panel.as_text())
        self._panel.note(f"Feed written to {path}", style="cyan")

    def action_clear_feed(self) -> None:
        """Clear the panel."""
        self._panel.clear_feed()


def main() -> None:
    """Entry point for `make tui`."""
    from customer_support_agent.core import build_agent

    try:
        AgenticTuiApp(lambda reporter: build_agent(reporter=reporter)).run()
    except KeyboardInterrupt:
        pass
