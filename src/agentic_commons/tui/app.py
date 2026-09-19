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

from collections.abc import Sequence
from typing import Any, Callable
from uuid import uuid4

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Footer, Header, Input

from agentic_commons.commands import Command, CommandRegistry, ConsoleEffect
from agentic_commons.commands.data import MemoryCommand
from agentic_commons.config import get_settings
from agentic_commons.logging import get_logger
from agentic_commons.logging.trace_stream import TraceStream
from agentic_commons.observability import InMemoryLiveStatusFeed, StageReporter
from agentic_commons.observability.types import StatusEvent
from agentic_commons.tui.clipboard import Osc52ClipboardService
from agentic_commons.tui.command_view import CommandView
from agentic_commons.tui.labels import StageLabelRegistry
from agentic_commons.tui.suggest import CommandSuggester
from agentic_commons.tui.widgets import LiveStatusPanel, StatusLineWidget, TokenCounterWidget

# Builds the agent, given the reporter it should emit status through. A
# factory rather than a built agent: the app owns the feed, so the agent
# cannot be constructed until the app exists.
AgentFactory = Callable[[StageReporter], Any]

# What `/debug off` passes back as its requested level.
_TRACE_OFF = "off"

# Shown when no caller names the window. An episode passes its own.
_DEFAULT_TITLE = "Customer Support Agent"


class AgenticTuiApp(App[None]):
    """Live status feed over a running agent."""

    # Laid out like Claude Code: the transcript takes the space it can, the
    # prompt sits directly under it in a bordered box, and the spinner and
    # token counts run *below* the prompt where they read as a status on
    # the work in flight.
    #
    # Nothing is docked. `dock` pins a widget to the screen edge and takes
    # it out of the flow, which is what left the prompt stranded at the
    # very bottom with a gap above it — and, earlier, put three docked
    # widgets in the same place.
    CSS = """
    Screen { layout: vertical; }
    #feed { height: 1fr; padding: 0 1; }
    #query {
        height: auto;
        margin: 0 1;
        border: round $accent;
    }
    #query:focus { border: round $accent-lighten-2; }
    #status-line { height: auto; padding: 0 2; }
    #tokens { height: auto; padding: 0 2; }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+o", "copy_feed", "Copy feed"),
        Binding("ctrl+l", "clear_feed", "Clear"),
    ]

    def __init__(
        self,
        agent_factory: "AgentFactory | None" = None,
        title: str = "",
        subtitle: str = "",
        commands: "Sequence[Command] | None" = None,
    ) -> None:
        super().__init__()
        self.title = title or _DEFAULT_TITLE
        # Textual draws this beside the title. For an episode it is the
        # claim the episode has to earn, which is what a viewer should
        # have in front of them while they watch the trace.
        self.sub_title = subtitle
        self._feed = InMemoryLiveStatusFeed()
        # Kept, not just called: `/harness` changes which subsystems are
        # enabled, and the agent standing in front of us was wired under
        # the previous answer.
        self._factory = agent_factory
        self._agent = agent_factory(StageReporter(self._feed, "tui")) if agent_factory else None
        self._labels = StageLabelRegistry()
        self._copier = Osc52ClipboardService()
        self._panel = LiveStatusPanel(self._labels)
        self._status = StatusLineWidget()
        self._tokens = TokenCounterWidget()
        self._commands = CommandRegistry()
        self._view = CommandView()
        # Mirrors the log stream into the same panel. The files are
        # written regardless; this only decides what reaches the screen.
        self._trace = TraceStream(self._panel.show_line)
        self._turn = 0
        # One session for the life of the window. Keying it per turn would
        # give every message its own conversation, so a follow-up like
        # "what about 10010" would have nothing to refer to.
        self._session_id = f"tui-{uuid4().hex[:8]}"
        # No customer is bound here. The conversation authenticates:
        # the agent asks for a code and verifies it against the database,
        # which is the flow a real deployment has.
        self._journal = get_logger("tui")
        for command in commands or ():
            self._commands.register(command)
        self._bind_live_commands()

    def _bind_live_commands(self) -> None:
        """Point the commands that inspect live state at this session's agent.

        `/memory working` reads the running facade, not the database —
        the active window exists only in the process. A console with no
        agent keeps the default command, which says so rather than
        showing an empty window.
        """
        memory = getattr(self._agent, "memory", None)
        if memory is not None:
            self._commands.register(MemoryCommand(memory=memory, session_id=self._session_id))

    def compose(self) -> ComposeResult:
        """The single-panel layout, prompt above its own status line."""
        yield Header()
        yield VerticalScroll(self._panel, id="feed")
        yield Input(
            placeholder="Ask the support agent…  (/help for commands)",
            id="query",
            suggester=CommandSuggester(self._commands),
        )
        yield self._status
        yield self._tokens
        yield Footer()

    def on_mount(self) -> None:
        """Subscribe to the feed and start the spinner clock."""
        self._feed.subscribe(self._on_status)
        self.set_interval(get_settings().tui.refresh_interval_ms / 1000, self._status.tick)
        self.query_one("#query", Input).focus()
        self._status.bind_tokens(self._tokens)
        if get_settings().tui.trace.enabled:
            self._trace.enable()
        self._panel.note(
            "Ready. The agent will ask you to verify with user_code before it looks anything up. "
            "Type /help for console commands.",
            style="dim",
        )

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Run one line — a slash command, or a message to the agent."""
        query = event.value.strip()
        if not query:
            return
        event.input.value = ""
        if CommandRegistry.is_command(query):
            await self._run_command(query)
            return
        await self._run(query)

    async def _run_command(self, line: str) -> None:
        """Dispatch a slash command and draw its result.

        Commands never reach the agent: no model call, no turn counted,
        nothing added to the customer's conversation. An operator
        inspecting the system should not show up in the transcript the
        customer is having.
        """
        self._panel.show_lines([self._view.echo(f"\n{line}")])
        result = await self._commands.dispatch(line)
        self._apply(result.effect)
        if result.effect.clear_transcript:
            return
        self._panel.show_lines(self._view.render(result))
        self._scroll_to_latest()

    def _apply(self, effect: ConsoleEffect) -> None:
        """Perform the console-level effects a command asked for."""
        if effect.clear_transcript:
            self._panel.clear_feed()
        if effect.trace_level:
            if effect.trace_level == _TRACE_OFF:
                self._trace.disable()
            else:
                self._trace.enable(effect.trace_level)
        if effect.rebuild_agent:
            self._rebuild_agent()

    def _rebuild_agent(self) -> None:
        """Re-wire the agent from configuration as it now stands.

        The conversation is deliberately kept. The point of rebuilding
        mid-session is to ask the *same* question of a differently
        configured agent, and a cleared transcript would take the
        comparison off the screen at the moment it was made.
        """
        if self._factory is None:
            return
        self._agent = self._factory(StageReporter(self._feed, "tui"))
        self._bind_live_commands()

    async def _run(self, query: str) -> None:
        """Execute a query, keeping the feed live throughout."""
        self._turn += 1
        self._tokens.start_turn()
        self._panel.note(f"\n> {query}", style="bold")
        self._scroll_to_latest()
        self._status.begin(self._labels.label_for("intake"))
        if self._agent is None:
            self._panel.note("No agent wired; feed demo only.", style="yellow")
            self._status.finish()
            return
        try:
            state = await self._agent.handle(query, session_id=self._session_id)
            self._reflect(state)
            self._panel.note(f"\n{state.response}", style="bright_green")
            self._scroll_to_latest()
        except Exception as exc:
            self._journal.exception("Agent run failed")
            self._panel.note(f"\nFailed: {exc}", style="red")
        finally:
            self._status.finish()

    def _reflect(self, state: Any) -> None:
        """Surface what the harness decided, beyond the stages it ran.

        The feed shows *what happened*; these lines show *what was
        decided* — which intent was matched, whether an action was
        refused, whether the reply was grounded. An operator watching a
        refusal needs the reason, not just the mark.
        """
        for decision in getattr(state, "decisions", []):
            style = "dim"
            if "refused" in decision or "deny" in decision:
                style = "red"
            elif "escalate" in decision or "suppressed" in decision:
                style = "yellow"
            elif decision.startswith(("intent", "scope", "auth")):
                style = "bright_magenta"
            elif "grounding" in decision:
                style = "orange3"
            self._panel.note(f"    · {decision}", style=style)

    def _on_status(self, event: StatusEvent) -> None:
        """Feed subscriber: update the panel, status line and counters."""
        self._panel.show_status(event)
        self._scroll_to_latest()
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

    def on_unmount(self) -> None:
        """Detach the trace stream so the handler does not outlive the app."""
        self._trace.disable()

    def action_clear_feed(self) -> None:
        """Clear the panel."""
        self._panel.clear_feed()

    def _scroll_to_latest(self) -> None:
        """Keep the newest line in view as the feed grows.

        A transcript that silently scrolls out of view is the same defect
        as not printing it: the person is watching for the step that is
        happening now.
        """
        try:
            self.query_one("#feed", VerticalScroll).scroll_end(animate=False)
        except Exception:
            pass


def main() -> None:
    """Entry point for `make tui`."""
    from customer_support_agent.core import build_agent

    try:
        AgenticTuiApp(lambda reporter: build_agent(reporter=reporter)).run()
    except KeyboardInterrupt:
        pass
