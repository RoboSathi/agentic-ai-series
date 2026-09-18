"""Live status feed — the event stream behind the TUI's feed (LLD §20).

Observer pattern: harness stages `emit()` as they start and finish; the TUI
panel subscribes rather than polling. Emitting is deliberately cheap and
never raises, so a UI problem can never break a request.
"""

from typing import Callable, Iterator, Protocol, runtime_checkable

from agentic_commons.config import get_settings
from agentic_commons.logging import get_logger
from agentic_commons.observability.types import StatusEvent, StatusKind

Subscriber = Callable[[StatusEvent], None]


@runtime_checkable
class LiveStatusFeed(Protocol):
    """Publishes harness stage transitions to interested observers."""

    def emit(self, event: StatusEvent) -> None: ...

    def events_for(self, run_id: str) -> Iterator[StatusEvent]: ...


class InMemoryLiveStatusFeed:
    """In-process feed with a bounded backlog."""

    def __init__(self, max_events: int | None = None) -> None:
        self._max_events = max_events or get_settings().tui.max_feed_lines
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
        if len(self._events) > self._max_events:
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
    ) -> None:
        """Emit one stage event for this run."""
        self._feed.emit(
            StatusEvent(
                run_id=self._run_id,
                stage=stage,
                status=status,
                tokens_delta=tokens_delta,
                detail=detail,
            )
        )

    def started(self, stage: str, detail: str | None = None) -> None:
        self.emit(stage, "started", detail=detail)

    def completed(
        self, stage: str, tokens_delta: int | None = None, detail: str | None = None
    ) -> None:
        self.emit(stage, "completed", tokens_delta=tokens_delta, detail=detail)

    def failed(self, stage: str, detail: str | None = None) -> None:
        self.emit(stage, "failed", detail=detail)
