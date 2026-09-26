"""Textual agent operations dashboard (LLD §20).

This TUI is intentionally a renderer, not an orchestrator and not a
logger. The graph and agent components decide what happens;
``StatusEvent`` objects are the semantic input contract. Logger records
remain in the logging subsystem and are not mirrored here.

It renders whatever agent it is given. Which agent that is arrives as an
``AgentHost`` — the title, the placeholder, the extra slash commands and
the stage labels a product needs all come from there, so this module
holds no product's vocabulary and can open against an agent it has never
heard of.
"""

from collections import Counter
from collections.abc import Sequence
from typing import Any, Callable
from uuid import uuid4

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, Input

from agentic_commons.commands import Command, CommandRegistry, ConsoleEffect
from agentic_commons.commands.data import MemoryCommand
from agentic_commons.commands.durability import (
    CheckpointsCommand,
    ResumeCommand,
    ThreadsCommand,
)
from agentic_commons.commands.harness import ToolsCommand
from agentic_commons.commands.system import SessionCommand
from agentic_commons.config import get_settings
from agentic_commons.durability import DurableAgent
from agentic_commons.host import AgentHost, resolve_host
from agentic_commons.logging import get_logger
from agentic_commons.observability import InMemoryLiveStatusFeed, StageReporter
from agentic_commons.observability.types import StatusEvent
from agentic_commons.tui.clipboard import Osc52ClipboardService
from agentic_commons.tui.command_view import CommandView
from agentic_commons.tui.labels import StageLabelRegistry
from agentic_commons.tui.suggest import CommandSuggester
from agentic_commons.tui.widgets import LiveStatusPanel, StatusLineWidget, TokenCounterWidget

AgentFactory = Callable[[StageReporter], Any]
# Used only when the console is opened with no host at all — a test, or a
# developer poking at the shell. A real run always names one.
_DEFAULT_TITLE = "Agent Harness"


# Recall tiers that answered without an embedding model.
_DETERMINISTIC_TIERS = ("exact", "lexical")


class AgenticTuiApp(App[None]):
    """Live semantic execution dashboard over a running support agent."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #feed {
        height: 1fr;
        padding: 0 1;
        scrollbar-size-vertical: 1;
        scrollbar-color: $accent;
        scrollbar-color-hover: $accent-lighten-2;
    }

    .feed-note {
        margin: 0 0 1 0;
    }

    StageEventWidget,
    .stage-event {
        width: 100%;
        margin: 0 0 1 0;
        padding: 0;
    }

    /* Textual renders the clickable Collapsible title as a separate
       CollapsibleTitle child. Color that widget, not only the parent. */

    .stage-event > CollapsibleTitle {
        text-style: bold;
    }

    /* RUNNING */
    .stage-started > CollapsibleTitle,
    .stage-started > CollapsibleTitle:hover,
    .stage-started > CollapsibleTitle:focus {
        color: cyan;
        background: transparent;
    }

    /* SUCCESS */
    .stage-completed > CollapsibleTitle,
    .stage-completed > CollapsibleTitle:hover,
    .stage-completed > CollapsibleTitle:focus {
        color: #00FF87;
        background: transparent;
    }

    /* WARNING / WAITING */
    .stage-warning > CollapsibleTitle,
    .stage-warning > CollapsibleTitle:hover,
    .stage-warning > CollapsibleTitle:focus,
    .stage-waiting > CollapsibleTitle,
    .stage-waiting > CollapsibleTitle:hover,
    .stage-waiting > CollapsibleTitle:focus {
        color: yellow;
        background: transparent;
    }

    /* FAILURE / BLOCKED */
    .stage-failed > CollapsibleTitle,
    .stage-failed > CollapsibleTitle:hover,
    .stage-failed > CollapsibleTitle:focus,
    .stage-blocked > CollapsibleTitle,
    .stage-blocked > CollapsibleTitle:hover,
    .stage-blocked > CollapsibleTitle:focus {
        color: red;
        background: transparent;
    }

    /* A partial hallucination is still blocked, but the response contained
       some grounded claims. Keep the block semantics while distinguishing it
       from a completely unsupported response. */
    .stage-blocked.stage-hallucination-partial > CollapsibleTitle,
    .stage-blocked.stage-hallucination-partial > CollapsibleTitle:hover,
    .stage-blocked.stage-hallucination-partial > CollapsibleTitle:focus {
        color: yellow;
        background: transparent;
    }

    .stage-details {
        padding: 0 1 1 3;
        color: $text-muted;
        height: auto;
    }

    #query {
        height: auto;
        margin: 0 1;
        border: round $accent;
    }

    #query:focus {
        border: round $accent-lighten-2;
    }

    #status-line {
        height: auto;
        padding: 0 2;
    }

    #tokens {
        height: 0;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+o", "copy_feed", "Copy"),
        Binding("ctrl+l", "clear_feed", "Clear"),
        Binding("pageup", "page_up", "Page up", priority=True),
        Binding("pagedown", "page_down", "Page down", priority=True),
        Binding("ctrl+u", "page_up", "Page up", priority=True, show=False),
        Binding("ctrl+d", "page_down", "Page down", priority=True, show=False),
        Binding("ctrl+g", "follow_latest", "Live", priority=True),
    ]

    def __init__(
        self,
        agent_factory: AgentFactory | None = None,
        title: str = "",
        subtitle: str = "",
        commands: Sequence[Command] | None = None,
        host: AgentHost | None = None,
    ) -> None:
        super().__init__()
        self._host = host
        # An explicit factory still wins — that is how a test substitutes
        # a scripted agent without needing a host at all.
        if agent_factory is None and host is not None:
            agent_factory = lambda reporter: host.build(reporter)  # noqa: E731
        self.title = title or (host.title() if host else _DEFAULT_TITLE)
        self.sub_title = subtitle

        self._feed = InMemoryLiveStatusFeed()
        self._factory = agent_factory
        self._agent = (
            agent_factory(StageReporter(self._feed, "tui"))
            if agent_factory
            else None
        )

        self._labels = StageLabelRegistry(host.stage_labels() if host else None)
        self._copier = Osc52ClipboardService()
        self._panel = LiveStatusPanel(self._labels)
        self._status = StatusLineWidget()
        self._tokens = TokenCounterWidget()
        self._commands = CommandRegistry()
        self._view = CommandView()
        self._turn = 0
        self._session_id = f"tui-{uuid4().hex[:8]}"
        self._follow_latest = True
        self._history_pending = 0
        # Which recall tier answered, this session — the deterministic hit-rate.
        self._recall_tiers: Counter[str] = Counter()
        self._journal = get_logger("tui")

        # The agent's own commands, then anything this caller added.
        for command in (*(host.commands() if host else ()), *(commands or ())):
            self._commands.register(command)
        self._bind_live_commands()

    def _bind_live_commands(self) -> None:
        """Point the commands that read live state at the agent on screen.

        Re-run after every rebuild. A command holding the previous
        agent's registry would describe the harness the operator just
        switched off, which is the exact thing `/harness` exists to make
        visible.
        """
        memory = getattr(self._agent, "working_memory", None)
        if memory is not None:
            self._commands.register(
                MemoryCommand(
                    working_memory=memory,
                    session_id=self._session_id,
                )
            )
        tools = self._commands.get("tools")
        if isinstance(tools, ToolsCommand):
            tools.bind(getattr(self._agent, "registry", None))
        session = self._commands.get("session")
        if isinstance(session, SessionCommand):
            session.bind(self._host, self._agent)
        for name in ("threads", "resume", "checkpoints"):
            command = self._commands.get(name)
            if isinstance(command, (ThreadsCommand, ResumeCommand, CheckpointsCommand)):
                command.bind(self._host, self._agent)

    def compose(self) -> ComposeResult:
        yield Header()
        yield self._panel
        yield Input(
            placeholder=(
                self._host.placeholder() if self._host else "Ask the agent…  (/help for commands)"
            ),
            id="query",
            suggester=CommandSuggester(self._commands),
        )
        yield self._status
        yield self._tokens
        yield Footer()

    def on_mount(self) -> None:
        self._feed.subscribe(self._on_status)
        self.set_interval(
            get_settings().tui.refresh_interval_ms / 1000,
            self._status.tick,
        )
        self.query_one("#query", Input).focus()
        self._status.bind_tokens(self._tokens)
        self._panel.note(
            "Author = RoboSathi | robosathi.com",
            style="bold cyan",
        )

        self._panel.note(
            "Before I can look anything up, I need to verify your account.\n"
            "Please share your 7-character user code.",
            style="yellow",
        )
        self._announce_resumable_runs()

    def _announce_resumable_runs(self) -> None:
        """Say, on opening, which runs the last process left unfinished.

        A crash is invisible from a fresh console otherwise: the new
        session starts clean and the half-finished turn sits in the
        checkpoint file where nobody looks.
        """
        if not isinstance(self._agent, DurableAgent):
            return
        pending = self._agent.runs.resumable(get_settings().commands.default_limit)
        if pending:
            self._panel.note(
                f"↻ {len(pending)} run(s) can be resumed or are waiting on a reviewer — /threads",
                style="bold yellow",
            )

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        query = event.value.strip()
        if not query:
            return
        event.input.value = ""

        if CommandRegistry.is_command(query):
            await self._run_command(query)
            return

        # No request queue yet. Run the turn as a Textual worker so the UI event
        # loop remains free to repaint the dashboard while the agent is running.
        event.input.disabled = True
        self.run_worker(
            self._run(query),
            exclusive=True,
            name="agent-turn",
        )

    async def _run_command(self, line: str) -> None:
        self._panel.note(
            line,
            style="bold bright_cyan",
        )
        result = await self._commands.dispatch(line)
        self._apply(result.effect)

        if not result.effect.clear_transcript:
            self._panel.show_lines(
                self._view.render(result)
            )

        self._schedule_follow_latest()

    def _apply(self, effect: ConsoleEffect) -> None:
        """Do the things only the console can do.

        A command states an effect and this applies it; giving commands a
        handle on the widgets would make every command a UI component.

        The rebuild is the one that matters for `/harness`. Which
        subsystems get wired is decided when the agent is composed, so an
        agent already on screen was built under the previous answer — and
        would go on running the harness the operator just switched off.
        """
        if effect.clear_transcript:
            self.action_clear_feed()
        if effect.rebuild_agent:
            self._rebuild_agent()
        if effect.session_id and effect.session_id != self._session_id:
            self._session_id = effect.session_id
            self._panel.note(f"Session switched to {effect.session_id}", style="cyan")

    def _rebuild_agent(self) -> None:
        """Compose the agent again under the configuration as it now stands.

        The conversation carries across when the agent knows how to take
        one over. A rebuild changes which controls are wired, not who is
        on the other end — `/harness off` is meant to be asked the same
        question again, and an agent that had forgotten the customer
        would ask them to verify before every comparison.

        Asked for by name rather than required: the console composes
        whatever agent it was given, and one with no conversation to
        carry simply does not offer the method.
        """
        if self._factory is None:
            return
        previous = self._agent
        self._agent = self._factory(StageReporter(self._feed, "tui"))
        adopt = getattr(self._agent, "adopt_conversation", None)
        if previous is not None and callable(adopt):
            adopt(previous)
        self._bind_live_commands()


    async def _run(self, query: str) -> None:
        self._turn += 1
        self._tokens.start_turn()
        self._follow_latest = True
        self._history_pending = 0
        self._status.set_following(True)
        self._status.set_history_pending(0)

        self._panel.note(
            f"\n══════════════ TURN {self._turn:02d} ══════════════",
            style="bold cyan",
        )
        self._panel.note(
            f"USER  {query}",
            style="bold white",
        )

        self._schedule_follow_latest()
        self._status.begin(self._labels.title_for("intake"))

        if self._agent is None:
            self._panel.note(
                "No agent wired; dashboard feed only.",
                style="yellow",
            )
            self._status.fail()
            self.query_one("#query", Input).disabled = False
            self.query_one("#query", Input).focus()
            return

        try:
            state = await self._agent.handle(
                query,
                session_id=self._session_id,
            )

            self._panel.note("\nASSISTANT", style="bold green")
            self._panel.note(
                state.response or "",
                style="white",
            )
            self._panel.note(
                f"\n──────── TURN {self._turn:02d} COMPLETE · "
                f"{self._status.elapsed_seconds:.2f}s · "
                f"{_compact(self._tokens.turn_tokens)} tokens ────────",
                style="dim",
            )
            self._status.finish()
            self._schedule_follow_latest()

        except Exception as exc:
            self._journal.exception("Agent run failed")
            detail = str(exc) or type(exc).__name__
            self._panel.note(
                f"\n✗ RUN FAILED: {detail}",
                style="red",
            )
            self._status.fail()
            self._schedule_follow_latest()

        finally:
            query_input = self.query_one("#query", Input)
            query_input.disabled = False
            query_input.focus()

    def _track_context(self, event: StatusEvent) -> None:
        """Keep the status line's one functional statistic current.

        How often recall was answered without an embedding model, this
        session. Ids and checkpoint counts are operational and live in
        the log, not on screen.
        """
        if event.stage != "memory_recall":
            return
        self._recall_tiers[str(event.fields.get("answered_by") or "nothing")] += 1
        answered = sum(self._recall_tiers[tier] for tier in _DETERMINISTIC_TIERS)
        total = answered + self._recall_tiers["dense"]
        if total:
            self._status.set_context(
                f"recall without embeddings {round(100 * answered / total)}%"
            )

    def _on_status(self, event: StatusEvent) -> None:
        """Render one semantic event and update the current-stage status."""
        self._track_context(event)
        if event.status == "started":
            self._status.update_stage(
                self._labels.title_for(
                    event.stage,
                    tool_name=str(event.fields.get("tool", event.detail or "")),
                )
            )
        else:
            self._panel.show_status(event)

            # If a failure / warning / blocked event happens, leave the event
            # visible at once. Otherwise preserve the user's chosen history
            # position.
            if not self._follow_latest:
                self._history_pending += 1
                self._status.set_history_pending(self._history_pending)

        if event.tokens_delta:
            self._tokens.add(event.tokens_delta)

        if self._follow_latest:
            self._schedule_follow_latest()

    def reporter(self, run_id: str) -> StageReporter:
        return StageReporter(self._feed, run_id)

    def action_copy_feed(self) -> None:
        path = self._copier.copy(self._panel.as_text())
        self._panel.note(
            f"Feed written to {path}",
            style="cyan",
        )
        self._schedule_follow_latest()

    def action_clear_feed(self) -> None:
        self._panel.clear_feed()
        self._history_pending = 0
        self._status.set_history_pending(0)

    def action_page_up(self) -> None:
        self._set_following(False)
        self._panel.scroll_page_up(animate=False)

    def action_page_down(self) -> None:
        self._set_following(False)
        self._panel.scroll_page_down(animate=False)

    def action_follow_latest(self) -> None:
        self._set_following(True)
        self._schedule_follow_latest()

    def on_mouse_scroll(self, event: object) -> None:
        # Textual dispatches mouse / trackpad scroll events through the app.
        # Treat user wheel/trackpad interaction as an intentional history view.
        self._set_following(False)

    def _set_following(self, following: bool) -> None:
        self._follow_latest = following
        if following:
            self._history_pending = 0
        self._status.set_following(following)
        self._status.set_history_pending(self._history_pending)

    def _schedule_follow_latest(self) -> None:
        if not self._follow_latest:
            return
        # Mounting a Collapsible is queued by Textual.  Following on the next
        # refresh avoids racing the newly-mounted card's layout calculation.
        self.call_after_refresh(
            self._scroll_to_latest
        )

    def _scroll_to_latest(self) -> None:
        if not self._follow_latest:
            return
        try:
            self._panel.scroll_end(
                animate=False,
                immediate=True,
            )
        except Exception:
            pass


def _compact(tokens: int) -> str:
    return f"{tokens / 1000:.1f}k" if tokens >= 1000 else str(tokens)


def main() -> None:
    """Open the console against whatever agent the configuration names.

    The agent is resolved from `agent.host` rather than imported. That
    one line is the whole reason this package can be a platform: a
    second agent needs a config entry, not an edit here.
    """
    host = resolve_host(get_settings().agent.host)
    # Slow, native start-up work belongs before the event loop opens, not
    # on the first turn — see `AgentHost.prepare`.
    host.prepare()

    try:
        AgenticTuiApp(host=host).run()
    except KeyboardInterrupt:
        pass
