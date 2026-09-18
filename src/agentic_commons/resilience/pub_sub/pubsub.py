"""Publish/subscribe (LLD §13).

In-memory for now, but the interface is shaped for a broker-backed swap
later (E42): publish is fire-and-forget, subscribers are anonymous to the
publisher, and a handler that raises must not take the publisher down.
Those three properties are what make the eventual swap a change of
implementation rather than a change of every call site.
"""

from typing import Awaitable, Callable, Protocol, Union, runtime_checkable

from agentic_commons.logging import get_logger
from agentic_commons.resilience.types import Event

EventHandler = Union[Callable[[Event], None], Callable[[Event], Awaitable[None]]]


class Subscription:
    """A live subscription, cancellable by the subscriber that made it."""

    def __init__(self, topic: str, handler: EventHandler, unsubscribe: Callable[[], None]) -> None:
        self.topic = topic
        self.handler = handler
        self._unsubscribe = unsubscribe
        self._active = True

    @property
    def active(self) -> bool:
        return self._active

    def cancel(self) -> None:
        """Stop receiving events on this subscription."""
        if self._active:
            self._unsubscribe()
            self._active = False


@runtime_checkable
class PubSub(Protocol):
    """Topic-based fan-out between components that don't know each other."""

    async def publish(self, topic: str, event: Event) -> None: ...

    def subscribe(self, topic: str, handler: EventHandler) -> Subscription: ...


class InMemoryPubSub:
    """Synchronous in-process fan-out with isolated handler failures."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = {}
        self._logger = get_logger("resilience")

    def subscribe(self, topic: str, handler: EventHandler) -> Subscription:
        """Register a handler for one topic."""
        self._handlers.setdefault(topic, []).append(handler)
        return Subscription(topic, handler, lambda: self._remove(topic, handler))

    async def publish(self, topic: str, event: Event) -> None:
        """Deliver to every subscriber; one failing handler stops nothing.

        A publisher must not learn that a subscriber broke — that
        coupling is precisely what pub/sub exists to remove.
        """
        for handler in list(self._handlers.get(topic, [])):
            try:
                outcome = handler(event)
                if outcome is not None:
                    await outcome
            except Exception as exc:
                self._logger.error(
                    "Subscriber raised; continuing delivery", topic=topic, error=str(exc)
                )

    def subscriber_count(self, topic: str) -> int:
        """How many handlers are listening on a topic."""
        return len(self._handlers.get(topic, []))

    def _remove(self, topic: str, handler: EventHandler) -> None:
        self._handlers[topic] = [h for h in self._handlers.get(topic, []) if h is not handler]
