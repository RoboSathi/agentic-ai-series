"""Live, human-readable trace of the agent's execution.

The TUI has two observability layers:

* the stage feed (``StatusEvent``) answers *what high-level stage is active?*;
* this trace answers *what code just happened, with what inputs, and what came back?*

The file logs remain the authoritative, untruncated evidence. The terminal
gets a compact representation of every narrated call/return/skip/event while
``/debug`` is at ``DEBUG``. ``/debug info`` still provides the quieter view.

The key design rule is: **detailed does not mean raw**. We expose the
identifiers and counts that explain the flow, but still suppress payloads such
as full prompts, memory contents, assembled context and arbitrary PII.
"""

import logging
import re
from collections.abc import Mapping
from typing import Callable

from rich.text import Text

from agentic_commons.config import get_settings
from agentic_commons.config.settings import TraceStreamSettings
from agentic_commons.logging.core import SUBSYSTEM_COLORS, LoggerFactory
from agentic_commons.logging.narration import NARRATION, Narration

_IN = "in"
_OUT = "out"
_DETAIL = "·"
_LOGGER_PREFIX = "agentic."
_COUNT = re.compile(r"\d+ \w+")

_LEVEL_STYLES = {
    logging.DEBUG: "bright_black",
    logging.INFO: "white",
    logging.WARNING: "yellow",
    logging.ERROR: "bold red",
    logging.CRITICAL: "bold red",
}

_KIND_MARKS = {
    "call": ("→", "bright_blue"),
    "return": ("←", "bright_green"),
    "skip": ("↷", "yellow"),
    "event": ("•", "white"),
}


class TraceLineFormatter:
    """Render one ``Narration`` as a compact operator-facing trace line.

    At DEBUG every narrated call/return/skip/event is eligible for the live
    trace. The logger/handler level still controls the overall verbosity,
    so the existing ``/debug info`` and ``/debug off`` controls remain useful.
    """

    def __init__(self, settings: TraceStreamSettings | None = None) -> None:
        self._settings = settings or get_settings().tui.trace

    def wants(self, narration: Narration, level: int) -> bool:
        """Whether a record is useful at the current terminal verbosity.

        DEBUG is the exhaustive operator trace.

        INFO intentionally remains quieter:
        - WARNING/ERROR/CRITICAL are always shown.
        - INFO authored events can still be shown when ``show_events`` is on.
        - DEBUG lifecycle traces are suppressed.
        """
        if level >= logging.WARNING:
            return True

        if narration.kind in {"call", "return", "skip"}:
            return level >= logging.DEBUG

        if narration.kind == "event":
            return self._settings.show_events and level >= logging.INFO

        return False

    def format(self, narration: Narration, level: int) -> Text:
        """Render one event with a stable, scannable terminal layout.

        Examples::

            agent        → CALL screening the request  in "refund"
            agent        ← RETURN screening the request
                         out verdict=support · 0.1ms
            gatekeeper   • Checking authority
                         · decision=allow, amount=24.99
        """
        settings = self._settings
        label = settings.subsystem_labels.get(
            narration.subsystem,
            narration.subsystem,
        )
        mark, mark_style = _KIND_MARKS.get(
            narration.kind,
            ("•", "white"),
        )

        line = Text("  ")

        line.append(
            f"{label[:settings.subsystem_width]:<{settings.subsystem_width}}",
            style=SUBSYSTEM_COLORS.get(
                narration.subsystem,
                "white",
            ),
        )

        line.append(" ")
        line.append(mark, style=mark_style)
        line.append(" ")

        headline = self._headline(narration)

        if narration.kind == "call":
            line.append("CALL ", style="bold bright_blue")
        elif narration.kind == "return":
            line.append("RETURN ", style="bold bright_green")
        elif narration.kind == "skip":
            line.append("SKIP ", style="bold yellow")

        line.append(
            headline,
            style=_LEVEL_STYLES.get(level, "white"),
        )

        for marker, value in self._parts(narration):
            line.append(
                f"  {marker} ",
                style="bright_black",
            )
            line.append(
                value,
                style="white" if marker == _OUT else "bright_black",
            )

        if narration.kind == "return" and narration.elapsed_ms > 0:
            line.append(
                f"  · {narration.elapsed_ms:.1f}ms",
                style="bright_black",
            )

        return line

    def _headline(self, narration: Narration) -> str:
        """Choose human wording for the operation being observed."""
        if narration.kind == "event":
            headline = narration.headline
        else:
            headline = self._settings.steps.get(narration.qualname)

            if not headline:
                # IMPORTANT:
                # Do not hide an unconfigured instrumented call.
                # DEBUG should expose it automatically.
                headline = (
                    narration.qualname
                    .replace(".", " → ")
                    .replace("_", " ")
                )

        return (
            headline[:1].upper() + headline[1:]
            if headline
            else "event"
        )

    def _parts(self, narration: Narration) -> list[tuple[str, str]]:
        """Render useful inputs/outputs without dumping raw payloads."""

        if narration.kind == "event":
            values = self._values(
                narration.fields,
                named=True,
            )

            return [
                (_DETAIL, values)
            ] if values else []

        parts: list[tuple[str, str]] = []

        inputs = self._values(
            narration.arguments,
            named=True,
        )

        if inputs:
            parts.append(
                (_IN, inputs)
            )

        if (
            narration.kind in {"return", "skip"}
            and _worth_showing(narration.outcome)
        ):
            parts.append(
                (
                    _OUT,
                    self._outcome(narration.outcome),
                )
            )

        return parts

    def _outcome(self, value: str) -> str:
        """Render a return summary without hiding useful multi-field decisions."""

        flattened = " ".join(
            value.split()
        )

        # Multi-field summaries are deliberate operator metadata, for
        # example:
        #
        # action=support.process_refund
        # verdict=deny
        # reason=amount exceeds limit
        #
        # Do not run those through _bare(), because that would strip useful
        # labels. Give them a larger budget instead.
        if flattened.count("=") > 1:
            limit = max(
                self._settings.max_value_chars,
                100,
            )

            if len(flattened) <= limit:
                return flattened

            return f"{flattened[: limit - 1]}…"

        return self._bare(flattened)

    def _values(
        self,
        source: Mapping[str, object],
        named: bool,
    ) -> str:
        """Keep identifiers, decisions and counts; suppress content fields."""

        shown: list[str] = []

        for name, value in source.items():
            if name in self._settings.muted_fields:
                continue

            rendered = self._clip(
                str(value)
            )

            if not _worth_showing(rendered):
                continue

            if named:
                # If a summarized object already has safe fields, preserve
                # the nested structure:
                #
                # call[action=support.process_refund]
                #
                # rather than:
                #
                # call=action=support.process_refund
                shown.append(
                    f"{name}[{rendered}]"
                    if "=" in rendered
                    else f"{name}={rendered}"
                )
            else:
                shown.append(
                    self._bare(rendered)
                )

            if len(shown) >= self._settings.max_values:
                break

        return ", ".join(shown)

    def _bare(self, value: str) -> str:
        """Normalize a summarized value for a readable one-line display."""

        clipped = self._clip(value)

        # A single field=value summary is usually redundant because the
        # operation already supplies the context.
        #
        # Example:
        #
        #   result=success
        #
        # becomes:
        #
        #   success
        #
        # But multi-field decision summaries retain all field names.
        if clipped.count("=") == 1:
            clipped = clipped.split(
                "=",
                maxsplit=1,
            )[1]

        return (
            f'"{clipped}"'
            if " " in clipped and not _COUNT.fullmatch(clipped)
            else clipped
        )

    def _clip(self, value: str) -> str:
        """Flatten and bound one displayed value."""

        flattened = " ".join(
            value.split()
        )

        limit = self._settings.max_value_chars

        if limit <= 0 or len(flattened) <= limit:
            return flattened

        return f"{flattened[: max(1, limit - 1)]}…"


class LiveFeedLogHandler(logging.Handler):
    """Forward narrated log records to the TUI without breaking the app."""

    def __init__(
        self,
        sink: Callable[[Text], None],
        level: int = logging.DEBUG,
        formatter: TraceLineFormatter | None = None,
    ) -> None:
        super().__init__(level=level)
        self._sink = sink
        self._formatter = formatter or TraceLineFormatter()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.render(record)

            if line is not None:
                self._sink(line)

        except Exception:  # pragma: no cover
            # Observability must never be able to crash the agent.
            self.handleError(record)

    def render(
        self,
        record: logging.LogRecord,
    ) -> Text | None:
        """Render a record, falling back to a plain event for third parties."""

        narration = getattr(
            record,
            NARRATION,
            None,
        )

        if not isinstance(narration, Narration):
            narration = Narration(
                subsystem=record.name.removeprefix(
                    _LOGGER_PREFIX
                ),
                headline=record.getMessage(),
            )

        if not self._formatter.wants(
            narration,
            record.levelno,
        ):
            return None

        return self._formatter.format(
            narration,
            record.levelno,
        )


class TraceStream:
    """Manage one console's subscription to the structured log stream."""

    def __init__(
        self,
        sink: Callable[[Text], None],
    ) -> None:
        self._sink = sink
        self._handler: LiveFeedLogHandler | None = None

    @property
    def enabled(self) -> bool:
        return self._handler is not None

    @property
    def level(self) -> str:
        return get_settings().tui.trace.level.upper()

    def enable(
        self,
        level: str | None = None,
    ) -> None:
        """Start streaming at DEBUG by default, or at an explicit level."""

        self.disable()

        wanted = (
            level or self.level
        ).upper()

        numeric = (
            logging.getLevelNamesMapping().get(
                wanted,
                logging.DEBUG,
            )
        )

        self._handler = LiveFeedLogHandler(
            self._sink,
            level=numeric,
        )

        LoggerFactory.subscribe(
            self._handler
        )

    def disable(self) -> None:
        """Detach the live trace; file logging continues unchanged."""

        if self._handler is not None:
            LoggerFactory.unsubscribe(
                self._handler
            )
            self._handler = None


_EMPTY = {
    "",
    "None",
    "none",
    "0",
    "False",
    "[]",
    "{}",
    "0 items",
}


def _worth_showing(value: str) -> bool:
    return value not in _EMPTY