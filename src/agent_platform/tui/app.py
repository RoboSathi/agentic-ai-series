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

from collections.abc import Awaitable, Sequence
from typing import Any, Callable

from textual import events
from textual.actions import SkipAction
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Button, Footer, Header, OptionList, TextArea

from agent_platform.commands import (
    PREFIX,
    AbortCommand,
    Command,
    CommandRegistry,
    ConsoleEffect,
    ConsoleScope,
    ResumeCommand,
    ScopedCommand,
)
from agent_platform.commands.data import MemoryCommand
from agent_platform.config import get_settings
from agent_platform.durability import DurableAgent
from agent_platform.host import (
    ActionHost,
    AgentHost,
    CancellableHost,
    GreetingHost,
    HousekeepingHost,
    StatusHost,
    SubjectHost,
    resolve_host,
)
from agent_platform.llm.metering import metered
from agent_platform.logging import bound_channel, get_logger, new_session_id
from agent_platform.observability import InMemoryLiveStatusFeed, StageReporter
from agent_platform.observability.types import StatusEvent
from agent_platform.tui.clipboard import Osc52ClipboardService
from agent_platform.tui.command_view import CommandView
from agent_platform.tui.completion import TabCompleter
from agent_platform.tui.labels import StageLabelRegistry
from agent_platform.tui.widgets import (
    ActionBar,
    ChoiceMenu,
    LiveStatusPanel,
    StatusLineWidget,
    TokenCounterWidget,
    palette,
)

AgentFactory = Callable[[StageReporter], Any]
# Used only when the console is opened with no host at all — a test, or a
# developer poking at the shell. A real run always names one.
_DEFAULT_TITLE = "Agent Harness"


# Multi-line chat window
class ChatPrompt(TextArea):
    """Multiline chat editor: Enter submits, Shift+Enter inserts a newline."""

    async def _on_key(self, event: events.Key) -> None:
        # TextArea's handler is a coroutine: called without `await`, every other key was dropped.
        if event.key == "enter":
            event.prevent_default()
            submit = getattr(self.app, "action_submit_query", None)
            if submit is not None:
                submit()
            return
        if event.key == "shift+enter":
            # TextArea breaks a line only on plain Enter, which here submits.
            event.prevent_default()
            self.insert("\n")
            return
        await super()._on_key(event)


# Recall tiers that answered without an embedding model.


class AgenticTuiApp(App[None]):
    """Live semantic execution dashboard over a running support agent."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #feed {
        height: 1fr;
        padding: 0 2;
        scrollbar-size-vertical: 1;
        scrollbar-color: #B8945A;
        scrollbar-color-hover: #9A7340;
    }

    /* Feed is a timeline, not a stack of roomy cards. Keep semantic rows
       visually separated by the title backgrounds rather than empty lines. */
    .feed-note {
        margin: 0;
    }

    .feed-note.turn-banner {
        margin: 1 0 0 0;
    }

    .feed-note.turn-complete {
        margin: 0;
    }

    .feed-note.response-gap {
        height: 1;
        margin: 0;
        padding: 0;
    }

    .feed-note.agent-label {
        margin: 1 0 0 0;
    }

    StageEventWidget,
    .stage-event {
        width: 100%;
        margin: 0;
        padding: 0;
    }

    /* Textual renders the clickable Collapsible title as a separate
       CollapsibleTitle child. Color that widget, not only the parent. */

    .stage-event > CollapsibleTitle {
        text-style: bold;
    }

    /* The title's colour is set per card from `tui.outcome_colors` / `tui.group_colors`
       (an inline style, so hover and focus keep it); the CSS only keeps the title flat. */
    .stage-event > CollapsibleTitle,
    .stage-event > CollapsibleTitle:hover,
    .stage-event > CollapsibleTitle:focus {
        background: transparent;
    }

    .stage-details {
        padding: 0 1 0 3;
        color: $text-muted;
        height: auto;
    }

    #choices {
        height: auto;
        max-height: 8;
        margin: 0 1;
        border: round $secondary;
    }

    #actions {
        height: auto;
        margin: 0 1;
    }

    /* One row, no border: the variant's fill still says "button". */
    #actions Button {
        height: 1;
        border: none;
        padding: 0 1;
        margin-right: 1;
        min-width: 4;
        text-style: bold;
    }

    #actions Button:hover {
        text-style: bold reverse;
    }

    #actions Button:disabled {
        opacity: 50%;
        text-style: none;
    }

    #query {
        height: auto;
        min-height: 3;
        max-height: 10;
        margin: 0 1;
        border: round #6F5A35;
        background: #11100E;
        color: #F3EAD3;
        padding: 0 1;
        overflow-x: hidden;
        overflow-y: auto;
    }

    #query:focus {
        border: round #B8945A;
        background: #14120F;
        color: #F3EAD3;
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
        # ⌘C, where the terminal forwards it (Ghostty, Kitty, WezTerm, iTerm2
        # with CSI u). Priority, or the focused Input takes it for its own copy.
        Binding("super+c", "copy_selection", "Copy selection", priority=True, show=False),
        Binding("ctrl+l", "clear_feed", "Clear"),
        Binding("pageup", "page_up", "Page up", priority=True, show=False),
        Binding("pagedown", "page_down", "Page down", priority=True, show=False),
        Binding("ctrl+u", "page_up", "Page up", priority=True, show=False),
        Binding("ctrl+d", "page_down", "Page down", priority=True, show=False),
        # Complete a slash command; anything else keeps Tab's focus behaviour.
        Binding("tab", "complete(1)", "Complete", priority=True, show=False),
        Binding("shift+tab", "complete(-1)", "Complete back", priority=True, show=False),
        Binding("up", "menu_move(-1)", "Previous choice", show=False),
        Binding("down", "menu_move(1)", "Next choice", show=False),
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
        self._agent = agent_factory(StageReporter(self._feed, "tui")) if agent_factory else None

        self._labels = StageLabelRegistry(host.stage_labels() if host else None)
        self._copier = Osc52ClipboardService()
        self._panel = LiveStatusPanel(self._labels)
        self._status = StatusLineWidget()
        self._menu = ChoiceMenu()
        self._actions = ActionBar()
        self._tokens = TokenCounterWidget()
        self._commands = CommandRegistry()
        self._completer = TabCompleter(self._commands)
        self._view = CommandView()
        self._turn = 0
        identity = get_settings().agent
        self._session_id = new_session_id(
            identity.id, identity.session_separator, identity.session_hex_chars
        )
        self._follow_latest = True
        self._history_pending = 0
        self._journal = get_logger("tui")

        # Whose records a command may show: this session's verified subject,
        # when the agent can say who that is. Read live, so a session switch
        # moves every scoped command with it.
        self._scope = ConsoleScope(
            lambda: self._session_id,
            host.verified_subject if isinstance(host, SubjectHost) else None,
        )
        # The agent's own commands, then anything this caller added.
        for command in (*(host.commands() if host else ()), *(commands or ())):
            self._register_command(command)
        # The registry's own defaults (`/memory` among them) are scoped too.
        for command in self._commands.all_commands():
            if isinstance(command, ScopedCommand):
                command.bind_scope(self._scope)
        self._bind_live_commands()
        # Optional host hooks: absent, the console is exactly what it was.
        self._status_host = host if isinstance(host, StatusHost) else None
        self._action_host = host if isinstance(host, ActionHost) else None
        if isinstance(host, CancellableHost):
            # Priority: the focused input would otherwise take the key for itself.
            self._bindings.bind(host.cancel_key(), "cancel_run", "Cancel run", priority=True)

    def _bind_live_commands(self) -> None:
        """Point the commands that read live state at the agent on screen.

        Re-run after every rebuild. A command holding the previous
        agent's registry would describe the harness the operator just
        switched off, which is the exact thing `/harness` exists to make
        visible.
        """
        memory = getattr(self._agent, "working_memory", None)
        if memory is not None:
            self._register_command(
                MemoryCommand(working_memory=memory, session_id=self._session_id)
            )
        if isinstance(self._agent, DurableAgent):
            self._register_command(ResumeCommand(lambda: self._agent))
            self._register_command(AbortCommand(lambda: self._agent))

    def _register_command(self, command: Command) -> None:
        """Register a command, handing it the console's scope if it narrows by one."""
        if isinstance(command, ScopedCommand):
            command.bind_scope(self._scope)
        self._commands.register(command)

    def compose(self) -> ComposeResult:
        yield Header()
        yield self._panel
        yield self._status
        yield self._actions
        yield self._menu
        yield ChatPrompt(
            placeholder=(
                self._host.placeholder() if self._host else "Ask the agent…  (/help for commands)"
            ),
            id="query",
            soft_wrap=True,
            show_line_numbers=False,
        )
        yield self._tokens
        yield Footer()

    def on_mount(self) -> None:
        self._feed.subscribe(self._on_status)
        self.set_interval(
            get_settings().tui.refresh_interval_ms / 1000,
            self._status.tick,
        )
        self.query_one("#query", ChatPrompt).focus()
        self._status.bind_tokens(self._tokens)
        self._panel.note(
            "Author: RoboSathi | robosathi.com",
            style=palette("banner"),
        )
        greeting = self._greeting()
        if greeting:
            self._panel.note(greeting, style=palette("notice"))
        if self._status_host is not None or self._action_host is not None:
            self.set_interval(get_settings().tui.status_refresh_ms / 1000, self._refresh_status)
        if isinstance(self._host, HousekeepingHost):
            self.set_interval(self._host.housekeeping_interval(), self._housekeep)

    def _housekeep(self) -> None:
        """Run the host's clock-driven work, one run at a time, off the input path."""
        self.run_worker(
            self._run_housekeeping(), name="housekeeping", group="housekeeping", exclusive=True
        )

    async def _run_housekeeping(self) -> None:
        if not isinstance(self._host, HousekeepingHost):
            return
        try:
            for line in await self._host.housekeeping():
                self._panel.note(line, style=palette("notice"))
        except Exception as exc:
            self._journal.warning("Housekeeping failed", error=str(exc))

    def _greeting(self) -> str:
        """The agent's own first words, when it offers them; nothing otherwise.

        The platform has no words of its own to open with: what an agent
        asks first (a code, a question) is the agent's to say.
        """
        return self._host.greeting() if isinstance(self._host, GreetingHost) else ""

    def _refresh_status(self) -> None:
        """Redraw the status block and the action buttons from state the host already holds."""
        try:
            if self._status_host is not None:
                summary, detail = self._status_host.status_lines(self._session_id)
                self._status.set_block(summary, detail)
            if self._action_host is not None:
                busy = self.query_one("#query", ChatPrompt).disabled
                self._actions.offer(self._action_host.actions(self._session_id), busy)
        except Exception as exc:
            self._journal.warning("Status block failed to render", error=str(exc))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """A button runs its command exactly as if it had been typed."""
        command = self._actions.command_for(event.button.id)
        if command is not None:
            self.run_worker(self._run_command(command), name="action", group="action")

    def action_cancel_run(self) -> None:
        """The host's cancel key: ask the running turn to stop."""
        if isinstance(self._host, CancellableHost):
            stopped = self._host.cancel()
            self._panel.note(
                "⏹ cancel requested" if stopped else "Nothing is running.",
                style=palette("notice"),
            )

    def action_menu_move(self, step: int) -> None:
        self._menu.move(step)

    def action_complete(self, step: int) -> None:
        """Tab: complete the slash command being typed, cycling on repeat."""
        query = self.query_one("#query", ChatPrompt)
        completed = (
            self._completer.complete(query.text, step)
            if self.focused is query and not query.disabled
            else None
        )
        if completed is None:
            # Not a command (or nothing matches): Tab moves focus, as it always did.
            if step > 0:
                self.action_focus_next()
            else:
                self.action_focus_previous()
            return
        query.text = completed
        # After the completed word, so the next Tab completes the next argument.
        query.move_cursor(query.document.end)
        choices = self._completer.candidates(completed.rstrip())
        if len(choices) > 1:
            self.notify("  ".join(choice.rsplit(" ", maxsplit=1)[-1] for choice in choices))

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Typing is a free-text answer: the menu steps aside."""
        if event.text_area.text and self._menu.offered:
            self._menu.dismiss()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        value = self._menu.value_at(event.option_index)
        if value is not None and not self.query_one("#query", ChatPrompt).disabled:
            self._menu.dismiss()
            self._start_turn(value)

    def action_submit_query(self) -> None:
        """Submit the multiline prompt; Enter sends, Shift+Enter creates a newline."""
        query = self.query_one("#query", ChatPrompt).text.strip() or self._menu.chosen() or ""

        if not query:
            return

        self.query_one("#query", ChatPrompt).text = ""
        self._menu.dismiss()

        if CommandRegistry.is_command(query):
            self.run_worker(
                self._run_command(query),
                name="command",
            )
            return

        self._start_turn(query)

    def _start_turn(self, query: str) -> None:
        """Run a turn as a Textual worker, so the UI keeps repainting meanwhile."""
        self.query_one("#query", ChatPrompt).disabled = True
        self.run_worker(
            self._run(query),
            exclusive=True,
            name="agent-turn",
        )

    def _start_resume(self, thread_id: str) -> None:
        """Continue a stopped run as a turn: the same status line, cards and answer as any turn."""
        self.query_one("#query", ChatPrompt).disabled = True
        self.run_worker(
            self._turn_of(
                get_settings().tui.resume_banner, lambda: self._live_agent().resume(thread_id)
            ),
            exclusive=True,
            name="agent-turn",
        )

    async def _run_command(self, line: str) -> None:
        """Run a slash command; one that becomes a turn is shown once, by the turn.

        A conversational command (one that declares `forwards_turn`) is sent to
        the agent as a turn, and the turn's banner already prints `USER /<command> …`.
        Echoing the line and the command's title as well put it on screen three
        times.
        """
        # Slash-command output is new console output: return the feed to live
        # mode before dispatch so the result is followed even when the operator
        # had previously scrolled through history.
        self._set_following(True)
        forwarded = self._forwards_turn(line)
        # Slash commands are console operations, not agent turns. Clear the
        # previous turn's status so a stale "TURN COMPLETE" is not shown
        # underneath command output. A command that forwards/resumes a turn
        # will immediately establish fresh turn status in _turn_of().
        self._status.clear_turn()
        if not forwarded:
            self._panel.note(line, style=palette("command"))
        with bound_channel(get_settings().tui.channel):
            result = await self._commands.dispatch(line)
        self._apply(result.effect)

        effect = result.effect
        if not (effect.clear_transcript or effect.agent_turn or effect.resume_thread):
            self._panel.show_lines(self._view.render(result))

        self._schedule_follow_latest()

    def _forwards_turn(self, line: str) -> bool:
        """Whether the command named in this line declares that it becomes a turn."""
        words = line.strip().removeprefix(PREFIX).split()
        command = self._commands.get(words[0]) if words else None
        return bool(getattr(command, "forwards_turn", False))

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
            self._panel.note(f"Session switched to {effect.session_id}", style=palette("info"))
        if effect.agent_turn:
            self._start_turn(effect.agent_turn)
        if effect.resume_thread:
            self._start_resume(effect.resume_thread)

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
        await self._turn_of(
            f"USER  {query}",
            lambda: self._live_agent().handle(query, session_id=self._session_id),
        )

    def _live_agent(self) -> Any:
        """The agent a turn calls (`_turn_of` has already said so when there is none)."""
        return self._agent

    async def _turn_of(self, opening: str, call: Callable[[], Awaitable[Any]]) -> None:
        """One turn on screen: banner, opening line, live status, then the answer and footer.

        A typed message and a resumed run are both turns, and are shown the same way. A
        resume drawn as command output left the status line on its last stage (nothing
        started or finished it) and the answer in the command colour.
        """
        self._turn += 1
        self._tokens.start_turn()
        self._follow_latest = True
        self._history_pending = 0
        self._status.set_following(True)
        self._status.set_history_pending(0)

        self._panel.note(
            f"══════════════ TURN {self._turn:02d} ══════════════",
            style=palette("banner"),
            classes="turn-banner",
        )
        self._panel.note(opening, style=palette("user"))

        self._schedule_follow_latest()
        self._status.begin(self._labels.title_for("intake"))

        if self._agent is None:
            self._panel.note(
                "No agent wired; dashboard feed only.",
                style=palette("notice"),
            )
            self._status.fail()
            self.query_one("#query", ChatPrompt).disabled = False
            self.query_one("#query", ChatPrompt).focus()
            return

        try:
            # Every model call the turn makes is counted where it is made (the LLM
            # adapter); the turn only says where the count goes.
            with bound_channel(get_settings().tui.channel), metered(self._tokens.add):
                state = await call()

            self._panel.note(
                "AGENT",
                style=palette("assistant"),
                classes="agent-label",
            )
            self._panel.response(state.response or "")
            self._panel.note(" ", classes="response-gap")
            self._menu.offer(getattr(state, "choices", None) or [])
            # self._panel.note(
            #     f"──────── TURN {self._turn:02d} COMPLETE · {self._status.metrics()} ────────",
            #     style=palette("metrics"),
            #     classes="turn-complete",
            # )
            self._status.finish()
            self._schedule_follow_latest()

        except Exception as exc:
            self._journal.exception("Agent run failed")
            detail = str(exc) or type(exc).__name__
            self._panel.note(
                f"\n✗ RUN FAILED: {detail}",
                style=palette("error"),
            )
            self._status.fail()
            self._schedule_follow_latest()

        finally:
            query_input = self.query_one("#query", ChatPrompt)
            query_input.disabled = False
            query_input.focus()

    def _on_status(self, event: StatusEvent) -> None:
        """Render one semantic event and update the current-stage status.

        A started event names the stage on the status line; the panel draws a pending card
        for it only when its kind is in `tui.pending_card_kinds`.
        """
        self._panel.show_status(event)
        if event.status == "started":
            self._status.update_stage(
                self._labels.title_for(
                    event.stage,
                    tool_name=str(event.fields.get("tool", event.detail or "")),
                )
            )
        elif not self._follow_latest:
            # If a failure / warning / blocked event happens, leave the event
            # visible at once. Otherwise preserve the user's chosen history
            # position.
            self._history_pending += 1
            self._status.set_history_pending(self._history_pending)

        if self._follow_latest:
            self._schedule_follow_latest()

    def reporter(self, run_id: str) -> StageReporter:
        return StageReporter(self._feed, run_id)

    def action_copy_feed(self) -> None:
        path = self._copier.copy(self._panel.as_text())
        self._panel.note(
            f"Feed written to {path}",
            style=palette("info"),
        )
        self._schedule_follow_latest()

    def on_text_selected(self, _event: events.TextSelected) -> None:
        """Copy a mouse selection the moment it is made.

        Apple Terminal keeps ⌘C for itself and never sends it here, so
        waiting for a key would mean the copy never happens. Copying on
        select makes "select, ⌘C, ⌘V" work anyway: the clipboard already
        holds the text by the time ⌘V is pressed.
        """
        self._copy_selection()

    def action_copy_selection(self) -> None:
        """⌘C: copy the screen's selection, or leave the key to the Input."""
        if not self._copy_selection():
            raise SkipAction()

    def _copy_selection(self) -> bool:
        text = self.screen.get_selected_text()
        if not text:
            return False
        if self._copier.copy_text(text):
            self.notify(
                f"Copied {len(text)} characters", timeout=get_settings().tui.copy_toast_seconds
            )
        else:
            self.notify("Copy failed — no clipboard command found", severity="warning")
        return True

    def action_clear_feed(self) -> None:
        self._panel.clear_feed()
        self._history_pending = 0
        self._status.set_history_pending(0)

    def action_page_up(self) -> None:
        self._set_following(False)
        self._panel.scroll_page_up(animate=False)

    def action_page_down(self) -> None:
        self._panel.scroll_page_down(animate=False)
        self.call_after_refresh(self._follow_if_at_bottom)

    def on_mouse_scroll(self, event: events.MouseEvent) -> None:
        # Mouse scroll events bubble to the App. Since the prompt is now a
        # vertically scrollable TextArea, scrolling inside it must not put the
        # feed into HISTORY mode. Only scrolls whose originating control is
        # the feed (or one of its descendants) affect feed following.
        control = event.control
        if control is None or self._panel not in control.ancestors_with_self:
            return

        # Treat feed scrolling as an intentional history view, and resume
        # following when it reaches the bottom again.
        self._set_following(False)
        self.call_after_refresh(self._follow_if_at_bottom)

    def _follow_if_at_bottom(self) -> None:
        """Reaching the end of the feed is the way back to live: no key to remember."""
        if self._panel.scroll_y >= self._panel.max_scroll_y - 1:
            self._set_following(True)

    def _set_following(self, following: bool) -> None:
        self._follow_latest = following
        if following:
            self._history_pending = 0
        self._status.set_following(following)
        self._status.set_history_pending(self._history_pending)

    def _schedule_follow_latest(self) -> None:
        if not self._follow_latest:
            return
        # Let Textual perform its own post-refresh deferral. In Textual 8.x,
        # scroll_end(immediate=False) deliberately waits for the refresh/layout
        # that updates max_scroll_y before calculating the end position.
        self._panel.scroll_end(animate=False)


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
