"""Live status feed — the event stream behind the TUI's feed (LLD §20).

Observer pattern: harness stages `emit()` as they start and finish; the TUI
panel subscribes rather than polling. Emitting is deliberately cheap and
never raises, so a UI problem can never break a request.
"""

from typing import Any, Callable, Iterator, Mapping, Protocol, runtime_checkable

from agentic_commons.config import get_settings
from agentic_commons.logging import get_logger
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
    """

    def __init__(self, feed: LiveStatusFeed, run_id: str) -> None:
        self._feed = feed
        self._run_id = run_id

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
    ) -> None:
        """Emit one semantic runtime event for this run.

        The reporter is deliberately UI-agnostic. It carries structured
        facts; the TUI decides whether to render them compactly or in
        expanded DEBUG form.
        """
        self._feed.emit(
            StatusEvent(
                run_id=self._run_id,
                stage=stage,
                status=status,
                event_kind=event_kind,
                tokens_delta=tokens_delta,
                detail=detail,
                summary=summary,
                fields=dict(fields or {}),
            )
        )

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
    ) -> None:
        self.emit(
            stage,
            "completed",
            tokens_delta=tokens_delta,
            detail=detail,
            event_kind=event_kind,
            summary=summary,
            fields=fields,
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
