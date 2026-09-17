"""Terminal UI app for interactively querying the agent and watching it run."""

import logging
import threading
import time
from pathlib import Path

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.timer import Timer
from textual.widgets import Footer, Header, Input, Static

from agentic_commons.config import get_settings
from agentic_commons.llm.client import LLMClient
from agentic_commons.logging.stub import SUBSYSTEM_COLORS
from agentic_commons.persistence.database import Database


class LogDisplay(Static):
    """Panel displaying live trace logs."""

    DEFAULT_CSS = """
    LogDisplay {
        width: 100%;
        height: 1fr;
        border: solid $primary;
        background: $surface;
        color: $text;
        overflow: auto;
    }
    """

    def __init__(
        self,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        self.logs: list[str] = []

    def add_log(self, text: str) -> None:
        """Append a new, complete log line."""
        self.logs.append(text)
        self._trim_and_render()

    def append_to_last_log(self, text: str) -> None:
        """Append text to the most recent log line (for streamed tokens)."""
        if not self.logs:
            self.logs.append(text)
        else:
            self.logs[-1] += text
        self._trim_and_render()

    def clear_logs(self) -> None:
        """Clear all logs."""
        self.logs = []
        self.update("")

    def _trim_and_render(self) -> None:
        if len(self.logs) > 200:
            self.logs = self.logs[-200:]
        self.update(self._render_logs())

    def _render_logs(self) -> str:
        """Render logs for display."""
        return "\n".join(self.logs)

    def render(self) -> str:
        """Render the log content."""
        return self._render_logs()


class LogHandler(logging.Handler):
    """Handler that sends logs to a LogDisplay widget, safely from any thread."""

    def __init__(self, log_display: LogDisplay) -> None:
        super().__init__(level=logging.INFO)
        self.log_display = log_display

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record to the display, safely from the app's own thread or any other."""
        try:
            msg = self.format(record)
            app = self.log_display.app
            # call_from_thread raises if called from the app's own thread, so
            # only route through it when logging from a background worker.
            if threading.get_ident() == app._thread_id:
                self.log_display.add_log(msg)
            else:
                app.call_from_thread(self.log_display.add_log, msg)
        except Exception:
            self.handleError(record)


class AgenticTUI(App[None]):
    """Interactive terminal UI for agentic commons."""

    CSS = """
    Screen {
        layout: vertical;
    }

    /* Header/Footer dock themselves (top/bottom) by default. #logs,
    #status-line and #input-field are NOT docked — they flow vertically in
    the remaining space between Header and Footer, in this order. Docking
    more than one of them to the same edge would make them overlap, since
    docking pins to an edge rather than stacking. */
    #logs {
        height: 1fr;
    }

    #status-line {
        height: 1;
        color: $text-muted;
    }

    #input-field {
        height: 3;
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+y", "copy_query", "Copy last query"),
        ("ctrl+l", "copy_logs", "Copy logs"),
    ]

    TITLE = "Agentic Commons - Customer Support Agent"
    SUB_TITLE = "Refunds, order status, and human-in-the-loop approvals"

    _SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    _SPINNER_INTERVAL_SECONDS = 0.1

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        database: Database | None = None,
    ) -> None:
        super().__init__()
        self.processing = False
        self._last_query = ""
        self._llm_client = llm_client
        self._database = database
        self._processing_started_at = 0.0
        self._response_chars = 0
        self._spinner_index = 0
        self._status_timer: Timer | None = None
        self._log_handlers: list[tuple[logging.Logger, LogHandler]] = []

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Header()
        yield LogDisplay(id="logs")
        yield Static("", id="status-line")
        yield Input(
            id="input-field",
            placeholder="Ask about an order, e.g. 'refund order 10001'...",
        )
        yield Footer()

    def on_mount(self) -> None:
        """Configure when the app mounts."""
        settings = get_settings()
        log_dir = Path(settings.logging.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        log_display = self.query_one("#logs", LogDisplay)
        log_display.add_log("═══ Agentic Commons - Interactive TUI ═══")
        log_display.add_log("")
        log_display.add_log("Instructions:")
        log_display.add_log("  • Ask about an order, e.g. 'what's the status of order 10001?'")
        log_display.add_log("  • System must finish before accepting new queries")
        log_display.add_log("  • Ctrl+Y copies your last query, Ctrl+L copies the full log")
        log_display.add_log("  • Ctrl+C exits")
        log_display.add_log("")

        self._setup_logging(log_display)
        self.query_one("#input-field", Input).focus()

    def on_unmount(self) -> None:
        """Remove the handlers this instance added to shared global loggers.

        Without this, every AgenticTUI construction leaves its LogHandler
        attached to the process-wide `agentic.*` loggers forever — later
        instances (or unrelated code, e.g. in tests) would keep triggering
        handlers pointing at an app that's no longer running.
        """
        for logger, handler in self._log_handlers:
            logger.removeHandler(handler)
        self._log_handlers.clear()

    def _setup_logging(self, log_display: LogDisplay) -> None:
        """Route every subsystem logger to the log display.

        Subsystem names come from SUBSYSTEM_COLORS (agentic_commons.logging)
        — the same single source of truth log_call already uses — instead
        of a second, separately hardcoded list that could drift from it.
        """
        for subsystem in SUBSYSTEM_COLORS:
            logger = logging.getLogger(f"agentic.{subsystem}")
            handler = LogHandler(log_display)
            handler.setFormatter(
                logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
            )
            logger.addHandler(handler)
            self._log_handlers.append((logger, handler))

    def action_copy_query(self) -> None:
        """Copy the last submitted query to the system clipboard."""
        log_display = self.query_one("#logs", LogDisplay)
        if not self._last_query:
            log_display.add_log("[yellow]No query to copy yet[/yellow]")
            return
        self.copy_to_clipboard(self._last_query)
        log_display.add_log(f"[cyan]Copied query to clipboard:[/cyan] {self._last_query}")

    def action_copy_logs(self) -> None:
        """Save the full visible log transcript to a file and try the clipboard too.

        Markup tags (e.g. "[bright_green]...[/bright_green]") are stripped
        first — useful for pasting the transcript into a bug report.

        Terminal clipboard copy (OSC 52) has real, well-known size limits
        that vary by terminal — a transcript of any length can silently
        fail to actually reach the clipboard even though nothing raises an
        error. The file write always works and is the dependable path;
        the clipboard copy is attempted as a convenience on top.
        """
        log_display = self.query_one("#logs", LogDisplay)
        if not log_display.logs:
            log_display.add_log("[yellow]No logs to copy yet[/yellow]")
            return

        plain_text = "\n".join(Text.from_markup(line).plain for line in log_display.logs)

        log_dir = Path(get_settings().logging.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        export_path = log_dir / f"tui_log_export_{int(time.time())}.txt"
        export_path.write_text(plain_text)

        self.copy_to_clipboard(plain_text)
        log_display.add_log(
            f"[cyan]Saved {len(log_display.logs)} log lines to {export_path}[/cyan] "
            "[dim](also tried the clipboard, but terminal clipboard support varies)[/dim]"
        )

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission."""
        log_display = self.query_one("#logs", LogDisplay)

        if self.processing:
            log_display.add_log("[yellow]⚠ System is still processing. Please wait.[/yellow]")
            return

        query = event.value.strip()
        if not query:
            log_display.add_log("[yellow]Type a question first — e.g. an order id and what you need.[/yellow]")
            return
        self._last_query = query

        log_display.add_log("")
        log_display.add_log("[bold bright_green]▶ Processing Query[/bold bright_green]")
        log_display.add_log(f"[cyan]Query: {query}[/cyan]")
        log_display.add_log("")

        self.processing = True
        input_field = self.query_one("#input-field", Input)
        input_field.disabled = True

        self._processing_started_at = time.monotonic()
        self._response_chars = 0
        self._spinner_index = 0
        self._update_status()
        self._status_timer = self.set_interval(
            self._SPINNER_INTERVAL_SECONDS, self._update_status
        )

        self._run_pipeline(query)

    def _update_status(self) -> None:
        """Advance the spinner and refresh the live "<spinner> Accomplishing… (Ns)" line."""
        self._spinner_index = (self._spinner_index + 1) % len(self._SPINNER_FRAMES)
        frame = self._SPINNER_FRAMES[self._spinner_index]
        elapsed = time.monotonic() - self._processing_started_at
        self.query_one("#status-line", Static).update(f"{frame} Accomplishing… ({elapsed:.0f}s)")

    @work(thread=True, exclusive=True)
    def _run_pipeline(self, query: str) -> None:
        """Run the customer support agent off the UI thread so the UI stays responsive."""
        from customer_support_agent.core.graph import run_agent

        log_display = self.query_one("#logs", LogDisplay)

        try:
            response = run_agent(query, llm_client=self._llm_client, database=self._database)
            self._response_chars = len(response)
            self.call_from_thread(log_display.add_log, f"[cyan]Response:[/cyan] {response}")
        except Exception as exc:
            self.call_from_thread(log_display.add_log, f"[red]Error: {exc}[/red]")

        self.call_from_thread(self._on_pipeline_finished, log_display)

    def _on_pipeline_finished(self, log_display: LogDisplay) -> None:
        """Reset UI state once the pipeline finishes (always runs on the UI thread)."""
        self.processing = False
        if self._status_timer is not None:
            self._status_timer.stop()
            self._status_timer = None
        self.query_one("#status-line", Static).update("")

        input_field = self.query_one("#input-field", Input)
        input_field.disabled = False
        input_field.value = ""
        input_field.focus()

        elapsed = time.monotonic() - self._processing_started_at
        log_display.add_log("")
        log_display.add_log(
            f"[bold bright_green]✓ Processing Complete[/bold bright_green] "
            f"[dim]({elapsed:.1f}s · {self._response_chars} chars)[/dim]"
        )
        log_display.add_log("Ready for next query...")


def run_app() -> None:
    """Run the interactive Textual TUI app."""
    app = AgenticTUI()
    app.run()


if __name__ == "__main__":
    run_app()
