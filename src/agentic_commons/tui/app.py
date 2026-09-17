"""Terminal UI app for monitoring agentic commons execution."""

import logging

from textual.app import ComposeResult
from textual.widgets import Footer, Header, Input, Static

from agentic_commons.config import get_settings


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

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.logs: list[str] = []

    def add_log(self, text: str) -> None:
        """Add a log line to the display."""
        self.logs.append(text)
        if len(self.logs) > 200:
            self.logs = self.logs[-200:]
        self.update(self._render_logs())

    def clear_logs(self) -> None:
        """Clear all logs."""
        self.logs = []
        self.update("")

    def _render_logs(self) -> str:
        """Render logs for display."""
        return "\n".join(self.logs)

    def render(self) -> str:
        """Render the log content."""
        return self._render_logs()


class LogHandler(logging.Handler):
    """Handler that sends logs to LogDisplay widget."""

    def __init__(self, log_display: LogDisplay) -> None:
        super().__init__()
        self.log_display = log_display

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record to the display."""
        try:
            msg = self.format(record)
            self.log_display.add_log(msg)
        except Exception:
            self.handleError(record)


def run_app() -> None:
    """Run the interactive Textual TUI app."""
    from pathlib import Path

    from textual.app import App

    from customer_support_agent.dev.walking_skeleton import main as run_skeleton

    class AgenticTUI(App[None]):
        """Interactive terminal UI for agentic commons."""

        CSS = """
        Screen {
            layout: vertical;
        }

        Header {
            dock: top;
        }

        Footer {
            dock: bottom;
        }

        #input-field {
            dock: bottom;
            height: 3;
        }

        #logs {
            height: 1fr;
        }
        """

        BINDINGS = [
            ("ctrl+c", "quit", "Quit"),
        ]

        TITLE = "Agentic Commons - Interactive Query Processor"
        SUB_TITLE = "Phase 0 - Real-time Log Streaming"

        def __init__(self) -> None:
            super().__init__()
            self.log_display: LogDisplay | None = None
            self.input_field: Input | None = None
            self.processing = False

        def compose(self) -> ComposeResult:
            """Create child widgets."""
            yield Header()
            self.log_display = LogDisplay(id="logs")
            yield self.log_display
            self.input_field = Input(
                id="input-field",
                placeholder="Enter query or press Enter to run skeleton...",
            )
            yield self.input_field
            yield Footer()

        def on_mount(self) -> None:
            """Configure when the app mounts."""
            settings = get_settings()
            log_dir = Path(settings.logging.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

            if self.log_display:
                self.log_display.add_log("═══ Agentic Commons - Interactive TUI ═══")
                self.log_display.add_log("")
                self.log_display.add_log("Instructions:")
                self.log_display.add_log("  • Type a query or just press ENTER to run the walking skeleton")
                self.log_display.add_log("  • Logs will stream in real-time as the system processes")
                self.log_display.add_log("  • System must finish before accepting new queries")
                self.log_display.add_log("  • Press Ctrl+C to exit")
                self.log_display.add_log("")

            self._setup_logging()
            self.input_field.focus()

        def _setup_logging(self) -> None:
            """Setup logging to stream to the log display."""
            if not self.log_display:
                return

            for logger_name in [
                "agentic.config",
                "agentic.logging",
                "agentic.tui",
                "agentic.llm",
                "agentic.persistence",
                "agentic.observability",
                "agentic.audit",
                "agentic.evals",
                "agentic.guardrails",
                "agentic.resilience",
                "agentic.security",
                "agentic.memory",
                "agentic.tools",
                "agentic.gatekeeper",
                "agentic.hallucination_protection",
                "agentic.customer_support_agent",
            ]:
                logger = logging.getLogger(logger_name)
                handler = LogHandler(self.log_display)
                handler.setFormatter(
                    logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
                )
                logger.addHandler(handler)

        def on_input_submitted(self, event: Input.Submitted) -> None:
            """Handle input submission."""
            if self.processing:
                self.log_display.add_log("[yellow]⚠ System is still processing. Please wait.[/yellow]")
                return

            query = event.value

            if self.log_display:
                self.log_display.add_log("")
                self.log_display.add_log("[bold bright_green]▶ Processing Query[/bold bright_green]")
                if query:
                    self.log_display.add_log(f"[cyan]Query: {query}[/cyan]")
                else:
                    self.log_display.add_log("[cyan]Running Walking Skeleton...[/cyan]")
                self.log_display.add_log("")

            self.processing = True
            if self.input_field:
                self.input_field.disabled = True

            try:
                run_skeleton()
            except Exception as e:
                if self.log_display:
                    self.log_display.add_log(f"[red]Error: {e}[/red]")

            self.processing = False
            if self.input_field:
                self.input_field.disabled = False
                self.input_field.value = ""
                self.input_field.focus()

            if self.log_display:
                self.log_display.add_log("")
                self.log_display.add_log("[bold bright_green]✓ Processing Complete[/bold bright_green]")
                self.log_display.add_log("Ready for next query...")

    app = AgenticTUI()
    app.run()


if __name__ == "__main__":
    run_app()
