"""The live trace, narrated (LLD §20).

The curated stage feed says what the *agent* is doing — eight lines a
turn, in the customer's terms. This says what the *code* is doing: one
line per step, in the developer's terms, so the flow can be followed
while it happens.

It is deliberately not the file log. `logs/` keeps the evidence — full
payloads, exact arguments, whole return values — and a screen that
repeats it becomes unreadable at exactly the moment it is most needed.
So a record arrives here as a `Narration` and leaves as a sentence:

    agent      working out the intent     message=where is my order → name=order_enquiry
    context    assembling the window      task=where is my order → 4 items
    gatekeeper checking authority         action=support.process_refund → verdict=deny

Curated, not exhaustive. Narrating every instrumented call gave forty
lines a turn from eight modules — technically complete, practically
unreadable. So the screen shows the steps named in `tui.trace.steps`,
one line each, on return, so the line can say what went in and what came
out; plus author-written INFO lines, which are already sentences; plus
anything that failed. Everything else stays in `logs/`, in full.
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

# How the two halves of a step are labelled. Words rather than arrows:
# "in"/"out" is unambiguous in a terminal that may not render an arrow,
# and reads the same aloud as on screen.
_IN = "in"
_OUT = "out"
_DETAIL = "·"

# Loggers are named `agentic.<subsystem>`, which is what the factory
# builds them as.
_LOGGER_PREFIX = "agentic."

# A summarised collection: "4 items", "1 item".
_COUNT = re.compile(r"\d+ \w+")

_LEVEL_STYLES = {
    logging.DEBUG: "bright_black",
    logging.INFO: "white",
    logging.WARNING: "yellow",
    logging.ERROR: "bold red",
    logging.CRITICAL: "bold red",
}


class TraceLineFormatter:
    """Turns one `Narration` into the line a person reads.

    Separate from the handler so the wording is testable without a
    logging framework, and so the rules live in one place rather than
    spread across an `emit` method.
    """

    def __init__(self, settings: TraceStreamSettings | None = None) -> None:
        self._settings = settings or get_settings().tui.trace

    def wants(self, narration: Narration, level: int) -> bool:
        """Whether this record belongs on screen at all.

        Three things earn a line, and nothing else does:

        - a **curated step**, on return, so the line can say what went in
          and what came out;
        - an **author-written event** at INFO or above — already a
          sentence, written by whoever knew what mattered;
        - anything that **went wrong**.

        Everything else is in the files. This is the difference between
        a trace someone reads and a log file someone scrolls past: forty
        lines a turn from eight modules is technically complete and
        practically useless.
        """
        if level >= logging.WARNING:
            return True
        if narration.kind == "return":
            return narration.qualname in self._settings.steps
        if narration.kind == "event":
            return self._settings.show_events and level >= logging.INFO
        return False

    def format(self, narration: Narration, level: int) -> Text:
        """One step, in one standard shape.

            memory       Noting for this conversation   in current_intent  out stored

        Module, then what is happening, then what went in and what came
        back. Always those four parts, always in that order, always in
        those columns — a trace is read by scanning down one column, and
        a format that varies per line cannot be scanned at all.
        """
        settings = self._settings
        label = settings.subsystem_labels.get(narration.subsystem, narration.subsystem)
        line = Text("  ")
        line.append(
            f"{label[: settings.subsystem_width]:<{settings.subsystem_width}}",
            style=SUBSYSTEM_COLORS.get(narration.subsystem, "white"),
        )
        parts = self._parts(narration)
        headline = self._headline(narration)
        # Padded only when something follows it, so a step with nothing
        # to report does not end in a column of whitespace.
        line.append(
            f"{headline:<{settings.action_width}}" if parts else headline,
            style=_LEVEL_STYLES.get(level, "white"),
        )
        for marker, value in parts:
            line.append(f" {marker} ", style="bright_black")
            line.append(value, style="white" if marker == _OUT else "bright_black")
        return line

    def _headline(self, narration: Narration) -> str:
        """What is happening, as a sentence a person wrote.

        Never a method name. A step nobody gave words to does not belong
        on the screen — `wants()` has already dropped it — so there is no
        case here where the fallback is a signature.
        """
        headline = (
            narration.headline
            if narration.kind == "event"
            else self._settings.steps.get(narration.qualname, "")
        )
        return headline[:1].upper() + headline[1:] if headline else ""

    def _parts(self, narration: Narration) -> list[tuple[str, str]]:
        """The in/out pair for this step, whichever halves exist."""
        if narration.kind == "event":
            # An author's line is already a statement, so its fields are
            # detail rather than a return value: labelling them "out"
            # would claim the sentence was a function call.
            values = self._values(narration.fields, named=True)
            return [(_DETAIL, values)] if values else []
        parts: list[tuple[str, str]] = []
        inputs = self._values(narration.arguments, named=False)
        if inputs:
            parts.append((_IN, inputs))
        if _worth_showing(narration.outcome):
            parts.append((_OUT, self._bare(narration.outcome)))
        return parts

    def _values(self, source: Mapping[str, object], named: bool) -> str:
        """The values that matter, rendered the same way every time.

        A step's inputs and outputs are shown **bare**: the step already
        said what it was doing, so "Checking authority · in
        support.process_refund · out deny" reads as English, while
        "in action=support.process_refund out verdict=deny" reads as a
        stack frame.

        An author's own log fields keep their names, because the author
        chose them to mean something: "held 2, returned 2" is only
        intelligible with the names attached.
        """
        shown = [
            f"{name} {self._clip(str(value))}" if named else self._bare(str(value))
            for name, value in source.items()
            if name not in self._settings.muted_fields and _worth_showing(str(value))
        ]
        return ", ".join(shown[: self._settings.max_values])

    def _bare(self, value: str) -> str:
        """One value with no label, quoted when it is a phrase.

        A summarised object arrives as `topic=orders`; the field name was
        useful while choosing *which* attribute to show, and is noise
        once the step itself has been named, so it is dropped here.
        """
        clipped = self._clip(value)
        if "=" in clipped:
            clipped = clipped.split("=", maxsplit=1)[1]
        # Quoted only when it is something somebody wrote. A count is not
        # a phrase, and '"4 items"' reads as a quotation of nothing.
        return f'"{clipped}"' if " " in clipped and not _COUNT.fullmatch(clipped) else clipped

    def _clip(self, value: str) -> str:
        """One value, kept to a width that leaves the line readable."""
        flattened = " ".join(value.split())
        limit = self._settings.max_value_chars
        return flattened if len(flattened) <= limit else f"{flattened[: limit - 1]}…"


class LiveFeedLogHandler(logging.Handler):
    """Forwards narrated log records to a callback.

    Attached to every subsystem logger through `LoggerFactory.subscribe`,
    so a subsystem whose logger is built later still reaches the screen.
    """

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
        """Narrate one record onto the screen.

        `logging.Handler.emit` must never raise — a failure here would
        surface as an exception inside code that was merely logging.
        """
        try:
            line = self.render(record)
            if line is not None:
                self._sink(line)
        except Exception:  # pragma: no cover - defensive, per logging contract
            self.handleError(record)

    def render(self, record: logging.LogRecord) -> Text | None:
        """The line for one record, or None if it does not belong on screen.

        A record with no narration came from outside this harness — a
        library's logger that happens to be attached. It is shown as its
        own message rather than dropped, because silence about a
        third-party warning is worse than an unstyled line.
        """
        narration = getattr(record, NARRATION, None)
        if not isinstance(narration, Narration):
            narration = Narration(
                subsystem=record.name.removeprefix(_LOGGER_PREFIX),
                headline=record.getMessage(),
            )
        if not self._formatter.wants(narration, record.levelno):
            return None
        return self._formatter.format(narration, record.levelno)


class TraceStream:
    """Turns the trace feed on and off for one console.

    Owns the subscription so the app does not have to remember whether it
    is currently attached — `enable()` twice is the same as once, which
    matters when the toggle is a command an operator can hammer.
    """

    def __init__(self, sink: Callable[[Text], None]) -> None:
        self._sink = sink
        self._handler: LiveFeedLogHandler | None = None

    @property
    def enabled(self) -> bool:
        return self._handler is not None

    @property
    def level(self) -> str:
        """The configured level, as a name."""
        return get_settings().tui.trace.level.upper()

    def enable(self, level: str | None = None) -> None:
        """Start narrating log records onto the panel."""
        self.disable()
        wanted = (level or self.level).upper()
        self._handler = LiveFeedLogHandler(self._sink, level=logging.getLevelNamesMapping()[wanted])
        LoggerFactory.subscribe(self._handler)

    def disable(self) -> None:
        """Stop narrating. The files keep every record either way."""
        if self._handler is not None:
            LoggerFactory.unsubscribe(self._handler)
            self._handler = None


# Values that are the *absence* of a value. Printing them fills the line
# with words that mean "nothing happened here", which is what a blank
# already says.
_EMPTY = {"", "None", "none", "0", "False", "[]", "{}", "0 items"}


def _worth_showing(value: str) -> bool:
    """Whether a value tells the reader anything."""
    return value not in _EMPTY
