"""Live status feed — the event stream behind the TUI's feed (LLD §20).

Observer pattern: harness stages `emit()` as they start and finish; the TUI
panel subscribes rather than polling. Emitting is deliberately cheap and
never raises, so a UI problem can never break a request.
"""

import time
from typing import Any, Callable, Iterator, Mapping, Protocol, runtime_checkable

from agentic_commons.config import get_settings
from agentic_commons.logging import get_logger
from agentic_commons.observability.turn_tally import TurnTally
from agentic_commons.observability.types import EventKind, StatusEvent, StatusKind

Subscriber = Callable[[StatusEvent], None]


@runtime_checkable
class LiveStatusFeed(Protocol):
    """Publishes harness stage transitions to interested observers."""

    def emit(self, event: StatusEvent) -> None: ...

    def events_for(self, run_id: str) -> Iterator[StatusEvent]: ...


class InMemoryLiveStatusFeed:
    """In-process feed with a bounded backlog."""

    def __init__(self, max_events: int | None = None) -> None:
        configured = get_settings().tui.max_feed_lines
        self._max_events = configured if max_events is None else max_events
        self._events: list[StatusEvent] = []
        self._subscribers: list[Subscriber] = []
        self._logger = get_logger("observability")

    def subscribe(self, subscriber: Subscriber) -> Callable[[], None]:
        """Register an observer; returns a function that unsubscribes it."""
        self._subscribers.append(subscriber)

        def unsubscribe() -> None:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)

        return unsubscribe

    def emit(self, event: StatusEvent) -> None:
        """Record an event and notify observers; never raises at the call site."""
        self._events.append(event)
        if self._max_events > 0 and len(self._events) > self._max_events:
            self._events = self._events[-self._max_events :]
        for subscriber in list(self._subscribers):
            try:
                subscriber(event)
            except Exception as exc:
                self._logger.warning("Status feed subscriber failed", error=str(exc))

    def events_for(self, run_id: str) -> Iterator[StatusEvent]:
        """Every event recorded for one run, in order."""
        return iter([event for event in self._events if event.run_id == run_id])

    @property
    def events(self) -> list[StatusEvent]:
        """All buffered events (most recent last)."""
        return list(self._events)


class NullLiveStatusFeed:
    """Feed that discards events — the default when no UI is attached."""

    def emit(self, event: StatusEvent) -> None:
        return None

    def events_for(self, run_id: str) -> Iterator[StatusEvent]:
        return iter(())


class StageReporter:
    """Convenience wrapper that pairs started/completed emissions.

    Keeps call sites to one line each and guarantees a stage that starts is
    always closed out, including on failure.

    It also stamps `latency_ms` on every outcome — the time from a
    stage's `started` to its end, measured here once instead of by forty
    emitters that each forget — and writes every event, with the turn's
    operational context (thread, session), to the observability log. The
    cards show what happened; the ids that find it later live in the file.
    """

    def __init__(self, feed: LiveStatusFeed, run_id: str) -> None:
        self._feed = feed
        self._run_id = run_id
        self._context: dict[str, Any] = {}
        self._started: dict[str, float] = {}
        self._logger = get_logger("observability")
        self._tally: TurnTally | None = None
        self._turn_started = 0.0

    def begin_turn(self, **context: Any) -> None:
        """Bind log context (ids) until `end_turn`, and start counting."""
        self._context = {key: value for key, value in context.items() if value not in (None, "")}
        self._started.clear()
        self._tally = TurnTally()
        self._turn_started = time.monotonic()

    def end_turn(
        self,
        status: StatusKind = "completed",
        outcome: str = "answered",
        working_memory: Mapping[str, Any] | None = None,
        **log_fields: Any,
    ) -> None:
        """Emit the turn summary card; operational details go to the log only.

        `working_memory` is what the session's working memory holds now,
        already redacted: the summary shows it beside what this turn added.
        """
        tally = self._tally
        if tally is None:
            return
        self._tally = None
        seconds = time.monotonic() - self._turn_started
        self._logger.info(
            "Turn finished",
            outcome=outcome,
            **self._context,
            **log_fields,
            **({"working_memory": dict(working_memory)} if working_memory else {}),
        )
        self.emit(
            "turn_summary",
            status,
            event_kind="system",
            summary=tally.headline(outcome, seconds),
            fields=tally.fields(outcome, seconds, working_memory),
        )
        self._context = {}

    def emit(
        self,
        stage: str,
        status: StatusKind,
        tokens_delta: int | None = None,
        detail: str | None = None,
        *,
        event_kind: EventKind = "stage",
        summary: str | None = None,
        fields: Mapping[str, Any] | None = None,
        quiet: bool = False,
        group: str = "",
    ) -> None:
        """Emit one semantic runtime event for this run.

        The reporter is deliberately UI-agnostic. It carries structured
        facts; the TUI decides whether to render them compactly or in
        expanded DEBUG form.

        A `quiet` event is bookkeeping: it is logged and counted in the
        turn's summary but draws no card of its own, so a per-write or
        per-lookup record cannot crowd out what a person acts on.
        """
        event = StatusEvent(
            run_id=self._run_id,
            stage=stage,
            status=status,
            event_kind=event_kind,
            tokens_delta=tokens_delta,
            detail=detail,
            summary=summary,
            group=group,
            fields=self._stamped(stage, status, fields),
        )
        if self._tally is not None:
            self._tally.observe(event)
        self._logger.debug(
            "Stage event",
            stage=stage,
            status=status,
            summary=summary,
            fields=event.fields,
            **self._context,
        )
        if not quiet:
            self._feed.emit(event)

    def _stamped(
        self, stage: str, status: StatusKind, fields: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        """The event's own fields plus measured latency (ids stay in the log)."""
        stamped = {key: value for key, value in (fields or {}).items() if not _empty(value)}
        if status == "started":
            self._started[stage] = time.monotonic()
            return stamped
        began = self._started.pop(stage, None)
        if began is not None and "latency_ms" not in stamped:
            stamped["latency_ms"] = round((time.monotonic() - began) * 1000, 1)
        return stamped

    def started(
        self,
        stage: str,
        detail: str | None = None,
        *,
        event_kind: EventKind = "stage",
        summary: str | None = None,
        fields: Mapping[str, Any] | None = None,
    ) -> None:
        self.emit(
            stage,
            "started",
            detail=detail,
            event_kind=event_kind,
            summary=summary,
            fields=fields,
        )

    def completed(
        self,
        stage: str,
        tokens_delta: int | None = None,
        detail: str | None = None,
        *,
        event_kind: EventKind = "stage",
        summary: str | None = None,
        fields: Mapping[str, Any] | None = None,
        quiet: bool = False,
    ) -> None:
        self.emit(
            stage,
            "completed",
            tokens_delta=tokens_delta,
            detail=detail,
            event_kind=event_kind,
            summary=summary,
            fields=fields,
            quiet=quiet,
        )

    def failed(
        self,
        stage: str,
        detail: str | None = None,
        *,
        event_kind: EventKind = "stage",
        summary: str | None = None,
        fields: Mapping[str, Any] | None = None,
    ) -> None:
        self.emit(
            stage,
            "failed",
            detail=detail,
            event_kind=event_kind,
            summary=summary,
            fields=fields,
        )

    def warning(
        self,
        stage: str,
        detail: str | None = None,
        *,
        event_kind: EventKind = "system",
        summary: str | None = None,
        fields: Mapping[str, Any] | None = None,
    ) -> None:
        self.emit(
            stage,
            "warning",
            detail=detail,
            event_kind=event_kind,
            summary=summary,
            fields=fields,
        )

    def waiting(
        self,
        stage: str,
        detail: str | None = None,
        *,
        event_kind: EventKind = "stage",
        summary: str | None = None,
        fields: Mapping[str, Any] | None = None,
    ) -> None:
        self.emit(
            stage,
            "waiting",
            detail=detail,
            event_kind=event_kind,
            summary=summary,
            fields=fields,
        )

    def blocked(
        self,
        stage: str,
        detail: str | None = None,
        *,
        event_kind: EventKind = "decision",
        summary: str | None = None,
        fields: Mapping[str, Any] | None = None,
    ) -> None:
        self.emit(
            stage,
            "blocked",
            detail=detail,
            event_kind=event_kind,
            summary=summary,
            fields=fields,
        )


def _empty(value: Any) -> bool:
    """Whether a field prints nothing a reader can use: `None`, or an empty string or container.

    `False` and `0` are answers ("written = false", "dropped = 0") and stay.
    """
    return value is None or (isinstance(value, (str, list, tuple, dict, set)) and not value)
